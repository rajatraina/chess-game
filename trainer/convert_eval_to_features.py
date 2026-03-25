#!/usr/bin/env python3
"""
Convert Lichess eval JSONL.ZST into compact NNUE .features binary format.

Each output record stores:
- nnz (uint16): number of active one-hot feature indices
- indices (uint16[nnz]): active indices in [0, 782]
- target (float32): regression target already clamp/scale processed

Sharded mode (--shards + --num-shards): routes each example to a random shard with bounded
RAM during ingest, then shuffles records inside each shard file in a post-pass.

Example commands (repository root; data under trainer/localdata/). On macOS, prefix with
``caffeinate -i`` for long runs so the system stays awake:

    # Single compressed output file
    caffeinate -i python3 trainer/convert_eval_to_features.py \\
        --input trainer/localdata/lichess_db_eval.train.jsonl.zst \\
        --output trainer/localdata/lichess_db_eval.train.features.zst \\
        --progress-every 1000000

    # Sharded train (200 shards) and single-shard validation
    caffeinate -i python3 trainer/convert_eval_to_features.py \\
        --input trainer/localdata/lichess_db_eval.train.jsonl.zst \\
        --shards trainer/localdata/lichess_db_eval.train.features_shards \\
        --num-shards 200 --ingest-buffer-mb 1536 --progress-every 1000000

    caffeinate -i python3 trainer/convert_eval_to_features.py \\
        --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \\
        --shards trainer/localdata/lichess_db_eval.val.1M.features_shards \\
        --num-shards 1 --ingest-buffer-mb 512 --progress-every 200000

See trainer/README.md for the full training pipeline (split → convert → config → train_nnue).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np
import zstandard as zstd
import chess

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from trainer.nnue_data_loader import (
    FEATURES_HEADER_STRUCT,
    FEATURES_MAGIC,
    FEATURES_VERSION,
    _select_best_eval_entry,
    _parse_record_to_board_and_raw_cp,
    apply_cp_regression_target,
    iter_feature_sparse_records,
)
from trainer.nnue_model import NNUEFeatureExtractor

_RECORD_BYTES_OVERHEAD = 32


def _iter_jsonl_lines_zst(in_path: Path, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    with open(in_path, "rb") as fin:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fin) as reader:
            buffer = b""
            while True:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    nl = buffer.find(b"\n")
                    if nl == -1:
                        break
                    line = buffer[:nl]
                    buffer = buffer[nl + 1 :]
                    if line.strip():
                        yield line
            if buffer.strip():
                yield buffer


def _write_sparse_record(fout: Any, active_bytes: bytes, target: float) -> None:
    nnz = len(active_bytes) // 2
    fout.write(struct.pack("<H", nnz))
    fout.write(active_bytes)
    fout.write(struct.pack("<f", float(target)))


class ShardIngest:
    """Per-shard buffers capped so total RAM stays within ingest_buffer_mb."""

    def __init__(
        self,
        shards_dir: Path,
        num_shards: int,
        ingest_buffer_mb: float,
        compress: bool,
        compression_level: int,
        stack: contextlib.ExitStack,
    ) -> None:
        self.shards_dir = shards_dir
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.num_shards = num_shards
        ingest_bytes = int(float(ingest_buffer_mb) * 1024 * 1024)
        self.per_shard_cap = max(ingest_bytes // num_shards, 64 * 1024)
        self.compress = compress
        self.compression_level = int(compression_level)
        self.stack = stack
        self.buffers: List[List[Tuple[bytes, float]]] = [[] for _ in range(num_shards)]
        self.buf_bytes = [0] * num_shards
        self.fouts: List[Any] = [None] * num_shards

    def _shard_path(self, sid: int) -> Path:
        ext = ".features.zst" if self.compress else ".features"
        return self.shards_dir / f"shard_{sid:03d}{ext}"

    def _ensure_open(self, sid: int) -> None:
        if self.fouts[sid] is not None:
            return
        path = self._shard_path(sid)
        raw = self.stack.enter_context(open(path, "wb"))
        if self.compress:
            cctx = zstd.ZstdCompressor(level=self.compression_level, threads=-1)
            fout = self.stack.enter_context(cctx.stream_writer(raw))
        else:
            fout = raw
        fout.write(FEATURES_HEADER_STRUCT.pack(FEATURES_MAGIC, FEATURES_VERSION, 783, 0))
        self.fouts[sid] = fout

    def append(self, sid: int, active_bytes: bytes, target: float) -> None:
        self.buffers[sid].append((active_bytes, target))
        self.buf_bytes[sid] += len(active_bytes) + _RECORD_BYTES_OVERHEAD
        if self.buf_bytes[sid] >= self.per_shard_cap:
            self.flush_shard(sid)

    def flush_shard(self, sid: int) -> None:
        if not self.buffers[sid]:
            return
        self._ensure_open(sid)
        fout = self.fouts[sid]
        for ab, t in self.buffers[sid]:
            _write_sparse_record(fout, ab, t)
        self.buffers[sid].clear()
        self.buf_bytes[sid] = 0

    def close(self) -> None:
        for sid in range(self.num_shards):
            self.flush_shard(sid)


def shuffle_features_shard_inplace(path: Path, compress: bool, compression_level: int) -> int:
    """Shuffle records inside one .features / .features.zst file; returns number of records."""
    records = list(iter_feature_sparse_records(str(path)))
    if not records:
        if path.stat().st_size <= FEATURES_HEADER_STRUCT.size:
            path.unlink(missing_ok=True)
        return 0
    random.shuffle(records)
    tmp_path = path.parent / (path.name + ".shuffle_tmp")
    with open(tmp_path, "wb") as raw_out:
        if compress:
            cctx = zstd.ZstdCompressor(level=int(compression_level), threads=-1)
            with cctx.stream_writer(raw_out) as fout:
                fout.write(FEATURES_HEADER_STRUCT.pack(FEATURES_MAGIC, FEATURES_VERSION, 783, 0))
                for ab, t in records:
                    _write_sparse_record(fout, ab, t)
        else:
            fout = raw_out
            fout.write(FEATURES_HEADER_STRUCT.pack(FEATURES_MAGIC, FEATURES_VERSION, 783, 0))
            for ab, t in records:
                _write_sparse_record(fout, ab, t)
    os.replace(tmp_path, path)
    return len(records)


def convert_eval_jsonl_to_features_sharded(
    input_path: str,
    shards_dir: str,
    num_shards: int,
    ingest_buffer_mb: float,
    max_positions: Optional[int],
    cp_label_clip_min: float,
    cp_label_clip_max: float,
    cp_target_scale: float,
    progress_every: int,
    augment_color_symmetry: bool = True,
    compress_output: Optional[bool] = None,
    compression_level: int = 16,
) -> None:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    extractor = NNUEFeatureExtractor()
    in_path = Path(input_path)
    out_dir = Path(shards_dir)
    do_compress = bool(compress_output) if compress_output is not None else True

    kept = 0
    source_kept = 0
    augmented_kept = 0
    written_pos = 0
    written_neg = 0
    written_zero = 0
    seen = 0
    json_decode_errors = 0
    filtered_invalid = 0
    start_time = time.time()
    pos_bucket_edges = [0, 10, 50, 100, 200, 400, 700, 1000, 1500, 2000]
    pos_bucket_labels = [
        "0-10",
        "10-50",
        "50-100",
        "100-200",
        "200-400",
        "400-700",
        "700-1000",
        "1000-1500",
        "1500-2000",
        "white mates",
    ]
    neg_bucket_labels = [
        "-10-0",
        "-50--10",
        "-100--50",
        "-200--100",
        "-400--200",
        "-700--400",
        "-1000--700",
        "-1500--1000",
        "-2000--1500",
        "black mates",
    ]
    written_pos_bucket_counts = [0] * len(pos_bucket_labels)
    written_neg_bucket_counts = [0] * len(neg_bucket_labels)
    written_zero_bucket_count = 0

    if augment_color_symmetry and abs(cp_label_clip_min + cp_label_clip_max) > 1e-6:
        raise ValueError(
            "Color-symmetry augmentation requires symmetric cp clip bounds around 0, "
            f"got [{cp_label_clip_min}, {cp_label_clip_max}]"
        )

    def _rot180(square: int) -> int:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        return chess.square(7 - file_idx, 7 - rank_idx)

    def _build_color_symmetric_board(board: chess.Board) -> chess.Board:
        b2 = chess.Board(None)
        for sq, piece in board.piece_map().items():
            b2.set_piece_at(_rot180(sq), chess.Piece(piece.piece_type, not piece.color))
        b2.turn = not board.turn
        if board.has_kingside_castling_rights(chess.WHITE):
            b2.castling_rights |= chess.BB_A8
        if board.has_queenside_castling_rights(chess.WHITE):
            b2.castling_rights |= chess.BB_H8
        if board.has_kingside_castling_rights(chess.BLACK):
            b2.castling_rights |= chess.BB_A1
        if board.has_queenside_castling_rights(chess.BLACK):
            b2.castling_rights |= chess.BB_H1
        b2.ep_square = _rot180(board.ep_square) if board.ep_square is not None else None
        b2.halfmove_clock = board.halfmove_clock
        b2.fullmove_number = board.fullmove_number
        return b2

    def _bucket_index(abs_cp: float) -> int:
        for i in range(len(pos_bucket_edges) - 1):
            lo = pos_bucket_edges[i]
            hi = pos_bucket_edges[i + 1]
            if i == 0:
                if lo <= abs_cp <= hi:
                    return i
            else:
                if lo < abs_cp <= hi:
                    return i
        return len(pos_bucket_edges) - 2

    def _record_eval_kind(record: dict) -> Tuple[bool, Optional[float], Optional[int]]:
        best_eval = _select_best_eval_entry(record)
        pv0 = (best_eval.get("pvs") or [None])[0] if best_eval else None
        if pv0 is None:
            return False, None, None
        if "cp" in pv0:
            try:
                return True, float(pv0["cp"]), None
            except Exception:
                return False, None, None
        if "mate" in pv0:
            try:
                return True, None, int(pv0["mate"])
            except Exception:
                return False, None, None
        return False, None, None

    def _route_sample(active_bytes: bytes, target: float, ingest: ShardIngest) -> None:
        nonlocal kept, written_pos, written_neg, written_zero
        sid = random.randrange(num_shards)
        ingest.append(sid, active_bytes, target)
        kept += 1
        if target > 0:
            written_pos += 1
        elif target < 0:
            written_neg += 1
        else:
            written_zero += 1

    def _emit_from_board(board: chess.Board, raw_cp: float, record: dict, ingest: ShardIngest) -> None:
        nonlocal source_kept, augmented_kept, written_zero_bucket_count
        feats = extractor.board_to_features(board)
        active = np.flatnonzero(feats > 0.0).astype(np.uint16, copy=False)
        target = apply_cp_regression_target(
            raw_cp,
            cp_label_clip_min,
            cp_label_clip_max,
            cp_target_scale,
        )
        has_eval, cp_value, mate_value = _record_eval_kind(record)

        _route_sample(active.tobytes(), float(target), ingest)
        source_kept += 1
        if has_eval:
            if mate_value is not None:
                if mate_value > 0:
                    written_pos_bucket_counts[-1] += 1
                elif mate_value < 0:
                    written_neg_bucket_counts[-1] += 1
            elif cp_value is not None:
                cp_clamped = max(cp_label_clip_min, min(cp_label_clip_max, cp_value))
                if cp_clamped > 0:
                    written_pos_bucket_counts[_bucket_index(abs(cp_clamped))] += 1
                elif cp_clamped < 0:
                    written_neg_bucket_counts[_bucket_index(abs(cp_clamped))] += 1
                else:
                    nonlocal written_zero_bucket_count
                    written_zero_bucket_count += 1

        if augment_color_symmetry:
            sym_board = _build_color_symmetric_board(board)
            sym_feats = extractor.board_to_features(sym_board)
            sym_active = np.flatnonzero(sym_feats > 0.0).astype(np.uint16, copy=False)
            sym_target = -float(target)
            _route_sample(sym_active.tobytes(), sym_target, ingest)
            augmented_kept += 1
            if has_eval:
                if mate_value is not None:
                    if mate_value > 0:
                        written_neg_bucket_counts[-1] += 1
                    elif mate_value < 0:
                        written_pos_bucket_counts[-1] += 1
                elif cp_value is not None:
                    cp_clamped = max(cp_label_clip_min, min(cp_label_clip_max, cp_value))
                    sym_cp = -cp_clamped
                    if sym_cp > 0:
                        written_pos_bucket_counts[_bucket_index(abs(sym_cp))] += 1
                    elif sym_cp < 0:
                        written_neg_bucket_counts[_bucket_index(abs(sym_cp))] += 1
                    else:
                        written_zero_bucket_count += 1

    def _print_progress(final: bool = False) -> None:
        elapsed = max(1e-9, time.time() - start_time)
        seen_rate = seen / elapsed
        kept_rate = kept / elapsed
        prefix = "Final stats" if final else "Progress"
        print(
            f"{prefix}: seen={seen:,}, kept={kept:,}, filtered={filtered_invalid:,}, "
            f"json_errors={json_decode_errors:,}, keep_rate={((kept / seen) * 100.0 if seen else 0.0):.2f}%, "
            f"speed={seen_rate:,.0f} lines/s ({kept_rate:,.0f} kept/s)"
        )
        if augment_color_symmetry:
            print(
                f"  kept breakdown: source={source_kept:,}, augmented={augmented_kept:,}, "
                f"aug_factor={(kept / source_kept if source_kept else 0.0):.2f}x"
            )
        print(
            f"  written target signs: pos={written_pos:,}, neg={written_neg:,}, zero={written_zero:,}"
        )
        total_written_bucketed = (
            sum(written_pos_bucket_counts) + sum(written_neg_bucket_counts) + written_zero_bucket_count
        )
        print("  Written eval buckets after augmentation (% of all written examples):")
        zero_pct = (100.0 * written_zero_bucket_count / total_written_bucketed) if total_written_bucketed else 0.0
        print(f"    0 (exact): {zero_pct:.2f}% ({written_zero_bucket_count:,})")
        print("    positive:")
        for label, count in zip(pos_bucket_labels, written_pos_bucket_counts):
            pct = (100.0 * count / total_written_bucketed) if total_written_bucketed else 0.0
            print(f"    {label}: {pct:.2f}% ({count:,})")
        print("    negative:")
        for label, count in zip(neg_bucket_labels, written_neg_bucket_counts):
            pct = (100.0 * count / total_written_bucketed) if total_written_bucketed else 0.0
            print(f"    {label}: {pct:.2f}% ({count:,})")

    with contextlib.ExitStack() as stack:
        ingest = ShardIngest(
            out_dir,
            num_shards,
            ingest_buffer_mb,
            do_compress,
            compression_level,
            stack,
        )
        for line in _iter_jsonl_lines_zst(in_path):
            seen += 1
            try:
                record = json.loads(line.decode("utf-8"))
            except Exception:
                json_decode_errors += 1
                continue
            board, raw_cp = _parse_record_to_board_and_raw_cp(record)
            if board is None or raw_cp is None:
                filtered_invalid += 1
                continue
            _emit_from_board(board, raw_cp, record, ingest)
            if progress_every > 0 and seen > 0 and seen % progress_every == 0:
                _print_progress(final=False)
            if max_positions is not None and kept >= max_positions:
                break

        ingest.close()

    _print_progress(final=True)
    print(f"Ingest done. Kept {kept:,} training rows across up to {num_shards} shards under {out_dir}")

    print("Shuffling within each shard (post-pass)...")
    if do_compress:
        shard_files = sorted(out_dir.glob("shard_*.features.zst"))
    else:
        shard_files = sorted(out_dir.glob("shard_*.features"))
    total_shuffled = 0
    for sp in shard_files:
        if not sp.name.startswith("shard_"):
            continue
        n = shuffle_features_shard_inplace(sp, do_compress, compression_level)
        total_shuffled += n
        print(f"  {sp.name}: {n:,} records")
    print(f"Post-pass shuffle complete. Total records in shards: {total_shuffled:,}")


def convert_eval_jsonl_to_features(
    input_path: str,
    output_path: str,
    max_positions: Optional[int],
    cp_label_clip_min: float,
    cp_label_clip_max: float,
    cp_target_scale: float,
    progress_every: int,
    augment_color_symmetry: bool = True,
    compress_output: Optional[bool] = None,
    compression_level: int = 16,
) -> None:
    extractor = NNUEFeatureExtractor()
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    source_kept = 0
    augmented_kept = 0
    written_pos = 0
    written_neg = 0
    written_zero = 0
    seen = 0
    json_decode_errors = 0
    filtered_invalid = 0
    start_time = time.time()
    pos_bucket_edges = [0, 10, 50, 100, 200, 400, 700, 1000, 1500, 2000]
    pos_bucket_labels = [
        "0-10",
        "10-50",
        "50-100",
        "100-200",
        "200-400",
        "400-700",
        "700-1000",
        "1000-1500",
        "1500-2000",
        "white mates",
    ]
    neg_bucket_labels = [
        "-10-0",
        "-50--10",
        "-100--50",
        "-200--100",
        "-400--200",
        "-700--400",
        "-1000--700",
        "-1500--1000",
        "-2000--1500",
        "black mates",
    ]
    written_pos_bucket_counts = [0] * len(pos_bucket_labels)
    written_neg_bucket_counts = [0] * len(neg_bucket_labels)
    written_zero_bucket_count = 0
    shuffle_chunk_size = progress_every if progress_every and progress_every > 0 else 100000
    write_buffer: List[Tuple[bytes, float]] = []

    if augment_color_symmetry and abs(cp_label_clip_min + cp_label_clip_max) > 1e-6:
        raise ValueError(
            "Color-symmetry augmentation requires symmetric cp clip bounds around 0, "
            f"got [{cp_label_clip_min}, {cp_label_clip_max}]"
        )

    def _rot180(square: int) -> int:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        return chess.square(7 - file_idx, 7 - rank_idx)

    def _build_color_symmetric_board(board: chess.Board) -> chess.Board:
        """Create color-swapped + 180-rotated equivalent position."""
        b2 = chess.Board(None)
        for sq, piece in board.piece_map().items():
            b2.set_piece_at(_rot180(sq), chess.Piece(piece.piece_type, not piece.color))

        # Side to move swaps under color symmetry.
        b2.turn = not board.turn

        # Castling rights under 180-rotation + color swap:
        # WK->BQ, WQ->BK, BK->WQ, BQ->WK.
        if board.has_kingside_castling_rights(chess.WHITE):
            b2.castling_rights |= chess.BB_A8
        if board.has_queenside_castling_rights(chess.WHITE):
            b2.castling_rights |= chess.BB_H8
        if board.has_kingside_castling_rights(chess.BLACK):
            b2.castling_rights |= chess.BB_A1
        if board.has_queenside_castling_rights(chess.BLACK):
            b2.castling_rights |= chess.BB_H1

        b2.ep_square = _rot180(board.ep_square) if board.ep_square is not None else None
        b2.halfmove_clock = board.halfmove_clock
        b2.fullmove_number = board.fullmove_number
        return b2

    def _bucket_index(abs_cp: float) -> int:
        # abs_cp expected in [0, 2000] after clipping/mapping.
        for i in range(len(pos_bucket_edges) - 1):
            lo = pos_bucket_edges[i]
            hi = pos_bucket_edges[i + 1]
            if i == 0:
                if lo <= abs_cp <= hi:
                    return i
            else:
                if lo < abs_cp <= hi:
                    return i
        return len(pos_bucket_edges) - 2

    def _record_eval_kind(record: dict) -> Tuple[bool, Optional[float], Optional[int]]:
        """Return (has_eval, cp_value, mate_value) from the selected best PV."""
        best_eval = _select_best_eval_entry(record)
        pv0 = (best_eval.get("pvs") or [None])[0] if best_eval else None
        if pv0 is None:
            return False, None, None
        if "cp" in pv0:
            try:
                return True, float(pv0["cp"]), None
            except Exception:
                return False, None, None
        if "mate" in pv0:
            try:
                return True, None, int(pv0["mate"])
            except Exception:
                return False, None, None
        return False, None, None

    def _flush_buffer(fout: Any, final: bool = False) -> None:
        if not write_buffer:
            return
        random.shuffle(write_buffer)
        for active_bytes, target in write_buffer:
            _write_sparse_record(fout, active_bytes, target)
        if final:
            print(f"Flushed final shuffled chunk: {len(write_buffer):,} samples")
        else:
            print(f"Flushed shuffled chunk: {len(write_buffer):,} samples")
        write_buffer.clear()

    def _print_progress(final: bool = False) -> None:
        elapsed = max(1e-9, time.time() - start_time)
        seen_rate = seen / elapsed
        kept_rate = kept / elapsed
        prefix = "Final stats" if final else "Progress"
        print(
            f"{prefix}: seen={seen:,}, kept={kept:,}, filtered={filtered_invalid:,}, "
            f"json_errors={json_decode_errors:,}, keep_rate={((kept / seen) * 100.0 if seen else 0.0):.2f}%, "
            f"speed={seen_rate:,.0f} lines/s ({kept_rate:,.0f} kept/s)"
        )
        if augment_color_symmetry:
            print(
                f"  kept breakdown: source={source_kept:,}, augmented={augmented_kept:,}, "
                f"aug_factor={(kept / source_kept if source_kept else 0.0):.2f}x"
            )
        print(
            f"  written target signs: pos={written_pos:,}, neg={written_neg:,}, zero={written_zero:,}"
        )
        total_written_bucketed = (
            sum(written_pos_bucket_counts) + sum(written_neg_bucket_counts) + written_zero_bucket_count
        )
        print("  Written eval buckets after augmentation (% of all written examples):")
        zero_pct = (100.0 * written_zero_bucket_count / total_written_bucketed) if total_written_bucketed else 0.0
        print(f"    0 (exact): {zero_pct:.2f}% ({written_zero_bucket_count:,})")
        print("    positive:")
        for label, count in zip(pos_bucket_labels, written_pos_bucket_counts):
            pct = (100.0 * count / total_written_bucketed) if total_written_bucketed else 0.0
            print(f"    {label}: {pct:.2f}% ({count:,})")
        print("    negative:")
        for label, count in zip(neg_bucket_labels, written_neg_bucket_counts):
            pct = (100.0 * count / total_written_bucketed) if total_written_bucketed else 0.0
            print(f"    {label}: {pct:.2f}% ({count:,})")

    with contextlib.ExitStack() as stack:
        raw_out = stack.enter_context(open(out_path, "wb"))
        do_compress = bool(compress_output) if compress_output is not None else str(out_path).endswith(".zst")
        if do_compress:
            cctx = zstd.ZstdCompressor(level=int(compression_level), threads=-1)
            fout = stack.enter_context(cctx.stream_writer(raw_out))
        else:
            fout = raw_out

        fout.write(FEATURES_HEADER_STRUCT.pack(FEATURES_MAGIC, FEATURES_VERSION, 783, 0))
        for line in _iter_jsonl_lines_zst(in_path):
            seen += 1
            try:
                record = json.loads(line.decode("utf-8"))
            except Exception:
                json_decode_errors += 1
                continue
            board, raw_cp = _parse_record_to_board_and_raw_cp(record)
            if board is None or raw_cp is None:
                filtered_invalid += 1
                continue

            feats = extractor.board_to_features(board)
            active = np.flatnonzero(feats > 0.0).astype(np.uint16, copy=False)
            target = apply_cp_regression_target(
                raw_cp,
                cp_label_clip_min,
                cp_label_clip_max,
                cp_target_scale,
            )
            write_buffer.append((active.tobytes(), float(target)))
            kept += 1
            source_kept += 1
            if target > 0:
                written_pos += 1
            elif target < 0:
                written_neg += 1
            else:
                written_zero += 1
            has_eval, cp_value, mate_value = _record_eval_kind(record)
            if has_eval:
                if mate_value is not None:
                    if mate_value > 0:
                        written_pos_bucket_counts[-1] += 1
                    elif mate_value < 0:
                        written_neg_bucket_counts[-1] += 1
                elif cp_value is not None:
                    cp_clamped = max(cp_label_clip_min, min(cp_label_clip_max, cp_value))
                    if cp_clamped > 0:
                        written_pos_bucket_counts[_bucket_index(abs(cp_clamped))] += 1
                    elif cp_clamped < 0:
                        written_neg_bucket_counts[_bucket_index(abs(cp_clamped))] += 1
                    else:
                        written_zero_bucket_count += 1
            if augment_color_symmetry:
                sym_board = _build_color_symmetric_board(board)
                sym_feats = extractor.board_to_features(sym_board)
                sym_active = np.flatnonzero(sym_feats > 0.0).astype(np.uint16, copy=False)
                sym_target = -float(target)
                write_buffer.append((sym_active.tobytes(), sym_target))
                kept += 1
                augmented_kept += 1
                if sym_target > 0:
                    written_pos += 1
                elif sym_target < 0:
                    written_neg += 1
                else:
                    written_zero += 1
                if has_eval:
                    if mate_value is not None:
                        if mate_value > 0:
                            written_neg_bucket_counts[-1] += 1
                        elif mate_value < 0:
                            written_pos_bucket_counts[-1] += 1
                    elif cp_value is not None:
                        cp_clamped = max(cp_label_clip_min, min(cp_label_clip_max, cp_value))
                        sym_cp = -cp_clamped
                        if sym_cp > 0:
                            written_pos_bucket_counts[_bucket_index(abs(sym_cp))] += 1
                        elif sym_cp < 0:
                            written_neg_bucket_counts[_bucket_index(abs(sym_cp))] += 1
                        else:
                            written_zero_bucket_count += 1

            if progress_every > 0 and seen > 0 and seen % progress_every == 0:
                _flush_buffer(fout, final=False)
                _print_progress(final=False)
            if max_positions is not None and kept >= max_positions:
                break

            if len(write_buffer) >= shuffle_chunk_size:
                _flush_buffer(fout, final=False)

        _flush_buffer(fout, final=True)

    _print_progress(final=True)
    print(f"Done. Wrote {kept:,} samples to {out_path}")


_CONVERT_EXAMPLES = """
Examples (run from repository root; data under trainer/localdata/). macOS: use
caffeinate -i before python3 on long jobs so the machine does not sleep.

  Single output file (.features.zst):
    caffeinate -i python3 trainer/convert_eval_to_features.py \\
      --input trainer/localdata/lichess_db_eval.train.jsonl.zst \\
      --output trainer/localdata/lichess_db_eval.train.features.zst \\
      --progress-every 1000000

  Sharded train (200 shards) + single-shard val:
    caffeinate -i python3 trainer/convert_eval_to_features.py \\
      --input trainer/localdata/lichess_db_eval.train.jsonl.zst \\
      --shards trainer/localdata/lichess_db_eval.train.features_shards \\
      --num-shards 200 --ingest-buffer-mb 1536 --progress-every 1000000
    caffeinate -i python3 trainer/convert_eval_to_features.py \\
      --input trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \\
      --shards trainer/localdata/lichess_db_eval.val.1M.features_shards \\
      --num-shards 1 --ingest-buffer-mb 512 --progress-every 200000

Full NNUE training steps: trainer/README.md
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert eval .jsonl.zst to compact .features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CONVERT_EXAMPLES,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input eval file, e.g. trainer/localdata/lichess_db_eval.train.jsonl.zst",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .features file when not using --shards (default: from input name)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Output directory for sharded .features; enables sharded mode (ignores --output)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards (e.g. 200 for large train, 1 for val); used only with --shards",
    )
    parser.add_argument(
        "--ingest-buffer-mb",
        type=float,
        default=1536.0,
        help="Total RAM budget (MB) for ingest buffers across all shards (default: 1536)",
    )
    parser.add_argument("--max-positions", type=int, default=None, help="Optional max kept rows")
    parser.add_argument("--cp-label-clip-min", type=float, default=-2000.0)
    parser.add_argument("--cp-label-clip-max", type=float, default=2000.0)
    parser.add_argument("--cp-target-scale", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument(
        "--augment-color-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add 180-rotated color-swapped counterpart with negated target (default: enabled)",
    )
    parser.add_argument(
        "--compress-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write compressed output stream (.zst). Default: auto from output suffix.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=16,
        help="Zstd level for compressed output (default: 16)",
    )
    args = parser.parse_args()

    if args.shards:
        convert_eval_jsonl_to_features_sharded(
            input_path=args.input,
            shards_dir=args.shards,
            num_shards=int(args.num_shards),
            ingest_buffer_mb=float(args.ingest_buffer_mb),
            max_positions=args.max_positions,
            cp_label_clip_min=float(args.cp_label_clip_min),
            cp_label_clip_max=float(args.cp_label_clip_max),
            cp_target_scale=float(args.cp_target_scale),
            progress_every=int(args.progress_every),
            augment_color_symmetry=bool(args.augment_color_symmetry),
            compress_output=args.compress_output,
            compression_level=int(args.compression_level),
        )
        return

    output = args.output
    if output is None:
        p = Path(args.input)
        name = p.name
        if name.endswith(".jsonl.zst"):
            output = str(p.with_name(name[: -len(".jsonl.zst")] + ".features"))
        elif name.endswith(".zst"):
            output = str(p.with_name(name[: -len(".zst")] + ".features"))
        else:
            output = str(p.with_suffix(".features"))

    convert_eval_jsonl_to_features(
        input_path=args.input,
        output_path=output,
        max_positions=args.max_positions,
        cp_label_clip_min=float(args.cp_label_clip_min),
        cp_label_clip_max=float(args.cp_label_clip_max),
        cp_target_scale=float(args.cp_target_scale),
        progress_every=int(args.progress_every),
        augment_color_symmetry=bool(args.augment_color_symmetry),
        compress_output=args.compress_output,
        compression_level=int(args.compression_level),
    )


if __name__ == "__main__":
    main()

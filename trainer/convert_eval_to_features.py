#!/usr/bin/env python3
"""
Convert Lichess eval JSONL.ZST into compact NNUE .features binary format.

Each output record stores:
- nnz (uint16): number of active one-hot feature indices
- indices (uint16[nnz]): active indices in [0, 782]
- target (float32): regression target already clamp/scale processed
"""

import argparse
import json
import struct
import sys
import time
import random
from pathlib import Path
from typing import Optional

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
)
from trainer.nnue_model import NNUEFeatureExtractor


def convert_eval_jsonl_to_features(
    input_path: str,
    output_path: str,
    max_positions: Optional[int],
    cp_label_clip_min: float,
    cp_label_clip_max: float,
    cp_target_scale: float,
    progress_every: int,
    augment_color_symmetry: bool = True,
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
    write_buffer: list[tuple[bytes, float]] = []

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

    def _record_eval_kind(record: dict) -> tuple[bool, Optional[float], Optional[int]]:
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

    def _flush_buffer(fout, final: bool = False) -> None:
        if not write_buffer:
            return
        random.shuffle(write_buffer)
        for active_bytes, target in write_buffer:
            nnz = len(active_bytes) // 2
            fout.write(struct.pack("<H", nnz))
            fout.write(active_bytes)
            fout.write(struct.pack("<f", float(target)))
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

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        fout.write(FEATURES_HEADER_STRUCT.pack(FEATURES_MAGIC, FEATURES_VERSION, 783, 0))
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fin) as reader:
            buffer = b""
            while True:
                chunk = reader.read(64 * 1024)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    nl = buffer.find(b"\n")
                    if nl == -1:
                        break
                    line = buffer[:nl]
                    buffer = buffer[nl + 1 :]
                    if not line.strip():
                        continue
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
                if max_positions is not None and kept >= max_positions:
                    break

            if (max_positions is None or kept < max_positions) and buffer.strip():
                seen += 1
                try:
                    record = json.loads(buffer.decode("utf-8"))
                    board, raw_cp = _parse_record_to_board_and_raw_cp(record)
                    if board is not None and raw_cp is not None:
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
                    else:
                        filtered_invalid += 1
                except Exception:
                    json_decode_errors += 1

            if len(write_buffer) >= shuffle_chunk_size:
                _flush_buffer(fout, final=False)

        _flush_buffer(fout, final=True)

    _print_progress(final=True)
    print(f"Done. Wrote {kept:,} samples to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert eval .jsonl.zst to compact .features")
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
        help="Output .features file (default: input with .features extension)",
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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()

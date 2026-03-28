"""
NNUE data loading pipeline for chess training.

This module streams the Lichess evaluation dataset and converts each record into:
- fixed White-oriented NNUE features
- a White-perspective centipawn regression target (mates mapped to cp clip extremes)
"""

import json
import zstandard as zstd
import torch
import chess
import numpy as np
import struct
import contextlib
import re
import random
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, Any, Iterator, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path to import chess_game modules
sys.path.append(str(Path(__file__).parent.parent))

from trainer.nnue_model import NNUEFeatureExtractor

FEATURES_MAGIC = b"NNUEFEAT"
FEATURES_VERSION = 1
FEATURES_HEADER_STRUCT = struct.Struct("<8sBHI")
# magic (8s), version (B), input_size (H), reserved flags (I)


def _is_features_file(path: str) -> bool:
    p = str(path)
    return p.endswith(".features") or p.endswith(".features.zst")


def _iter_feature_records(path: str) -> Iterator[Tuple[np.ndarray, float]]:
    """Yield (features, target) from compact binary .features files."""
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(path, "rb"))
        p = str(path)
        if p.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            reader = stack.enter_context(dctx.stream_reader(f))
        else:
            reader = f

        header = reader.read(FEATURES_HEADER_STRUCT.size)
        if len(header) != FEATURES_HEADER_STRUCT.size:
            raise ValueError(f"Invalid .features header in {path}")
        magic, version, input_size, _reserved = FEATURES_HEADER_STRUCT.unpack(header)
        if magic != FEATURES_MAGIC:
            raise ValueError(f"Invalid .features magic in {path}")
        if version != FEATURES_VERSION:
            raise ValueError(f"Unsupported .features version {version} in {path}")
        if input_size != 783:
            raise ValueError(f"Unexpected input size {input_size} in {path}; expected 783")

        while True:
            raw_nnz = reader.read(2)
            if not raw_nnz:
                break
            if len(raw_nnz) != 2:
                raise ValueError(f"Corrupt record header in {path}")
            (nnz,) = struct.unpack("<H", raw_nnz)
            idx_buf = reader.read(nnz * 2)
            if len(idx_buf) != nnz * 2:
                raise ValueError(f"Corrupt feature index payload in {path}")
            target_buf = reader.read(4)
            if len(target_buf) != 4:
                raise ValueError(f"Corrupt target payload in {path}")
            indices = np.frombuffer(idx_buf, dtype=np.uint16).astype(np.int32, copy=False)
            feats = np.zeros(783, dtype=np.float32)
            feats[indices] = 1.0
            (target,) = struct.unpack("<f", target_buf)
            yield feats, float(target)


def iter_feature_sparse_records(path: str) -> Iterator[Tuple[bytes, float]]:
    """Yield (indices_bytes, target) for each record; indices_bytes length is 2*nnz (uint16 LE)."""
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(path, "rb"))
        p = str(path)
        if p.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            reader = stack.enter_context(dctx.stream_reader(f))
        else:
            reader = f

        header = reader.read(FEATURES_HEADER_STRUCT.size)
        if len(header) != FEATURES_HEADER_STRUCT.size:
            raise ValueError(f"Invalid .features header in {path}")
        magic, version, input_size, _reserved = FEATURES_HEADER_STRUCT.unpack(header)
        if magic != FEATURES_MAGIC:
            raise ValueError(f"Invalid .features magic in {path}")
        if version != FEATURES_VERSION:
            raise ValueError(f"Unsupported .features version {version} in {path}")
        if input_size != 783:
            raise ValueError(f"Unexpected input size {input_size} in {path}; expected 783")

        while True:
            raw_nnz = reader.read(2)
            if not raw_nnz:
                break
            if len(raw_nnz) != 2:
                raise ValueError(f"Corrupt record header in {path}")
            (nnz,) = struct.unpack("<H", raw_nnz)
            idx_buf = reader.read(nnz * 2)
            if len(idx_buf) != nnz * 2:
                raise ValueError(f"Corrupt feature index payload in {path}")
            target_buf = reader.read(4)
            if len(target_buf) != 4:
                raise ValueError(f"Corrupt target payload in {path}")
            (target,) = struct.unpack("<f", target_buf)
            yield bytes(idx_buf), float(target)


def list_nnue_feature_shard_paths(directory: str) -> List[str]:
    """Sorted paths to *.features / *.features.zst in directory (non-recursive)."""
    root = Path(directory)
    if not root.is_dir():
        return []
    paths: List[Path] = []
    paths.extend(root.glob("*.features"))
    paths.extend(root.glob("*.features.zst"))
    # Exclude accidental matches like foo.features.bak if any; glob is exact suffix.
    return sorted(str(p.resolve()) for p in paths)


# --- Streaming helpers ---
def _select_best_eval_entry(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the strongest available evaluation for a position.

    Lichess stores multiple evaluations per position with different PV counts, depths,
    and node counts. For training we prefer the deepest available evaluation, using
    knodes and PV count as tie-breakers.
    """
    evals = data.get("evals") or []
    if not evals:
        return None

    return max(
        evals,
        key=lambda entry: (
            entry.get("depth", -1),
            entry.get("knodes", -1),
            len(entry.get("pvs") or []),
        ),
    )


def apply_cp_regression_target(
    raw_cp: float,
    clip_min: float,
    clip_max: float,
    cp_target_scale: float,
) -> float:
    """Clamp White-perspective cp, then divide by cp_target_scale for network target."""
    c = float(max(clip_min, min(clip_max, raw_cp)))
    scale = float(cp_target_scale) if cp_target_scale else 1.0
    if scale <= 0:
        scale = 1.0
    return c / scale


def _parse_record_to_board_and_raw_cp(data: Dict[str, Any]) -> Tuple[Optional[chess.Board], Optional[float]]:
    """Parse a Lichess eval record into a board and raw White-perspective cp.

    Mate rows are kept by mapping mate sign to cp clip extremes before label clamp:
    - mate > 0 -> positive cp (White winning)
    - mate < 0 -> negative cp (Black winning)
    """
    try:
        fen = data["fen"]
        board = chess.Board(fen)
    except Exception:
        return None, None

    try:
        best_eval = _select_best_eval_entry(data)
        pvs = best_eval.get("pvs") if best_eval else None
        pv0 = pvs[0] if pvs else None
        if pv0 is None:
            return None, None

        if "cp" in pv0:
            return board, float(pv0["cp"])
        if "mate" in pv0:
            mate = int(pv0["mate"])
            if mate > 0:
                return board, 2000.0
            if mate < 0:
                return board, -2000.0
            return None, None
        return None, None
    except Exception:
        return None, None


def _record_to_target_summary(
    data: Dict[str, Any],
    clip_min: float,
    clip_max: float,
    cp_target_scale: float,
) -> Optional[Dict[str, Any]]:
    """Create a debug-friendly summary for one labeled record (cp and mapped mates)."""
    board, raw_cp = _parse_record_to_board_and_raw_cp(data)
    if board is None or raw_cp is None:
        return None

    best_eval = _select_best_eval_entry(data)
    pv0 = (best_eval.get("pvs") or [None])[0] if best_eval else None
    if pv0 is None:
        return None

    cp_clamped = float(max(clip_min, min(clip_max, raw_cp)))
    reg_target = apply_cp_regression_target(raw_cp, clip_min, clip_max, cp_target_scale)
    return {
        "fen": board.fen(),
        "side_to_move": "white" if board.turn else "black",
        "selected_depth": best_eval.get("depth"),
        "selected_knodes": best_eval.get("knodes"),
        "raw_cp": pv0.get("cp"),
        "raw_mate": pv0.get("mate"),
        "cp_clamped": cp_clamped,
        "regression_target": reg_target,
    }


class NNUETrainIterableDataset(IterableDataset):
    """Streaming iterable dataset for large .zst training files."""

    def __init__(
        self,
        data_file: str,
        max_positions: Optional[int] = None,
        chunk_size: int = 64 * 1024,
        cp_label_clip_min: float = -2000.0,
        cp_label_clip_max: float = 2000.0,
        cp_target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.data_file = data_file
        self.max_positions = max_positions
        self.chunk_size = chunk_size
        self.cp_label_clip_min = cp_label_clip_min
        self.cp_label_clip_max = cp_label_clip_max
        self.cp_target_scale = cp_target_scale
        self.feature_extractor = NNUEFeatureExtractor()

    def __len__(self) -> int:
        """Return estimated length for DataLoader compatibility."""
        if self.max_positions is not None:
            return self.max_positions
        return 1000000

    def _line_iterator(self):
        with open(self.data_file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buffer = b""
                while True:
                    chunk = reader.read(self.chunk_size)
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

    def __iter__(self):
        data_path = Path(self.data_file)
        if data_path.is_dir():
            shard_paths = list_nnue_feature_shard_paths(self.data_file)
            if not shard_paths:
                return
            random.shuffle(shard_paths)
            yielded = 0
            for shard_path in shard_paths:
                if not _is_features_file(shard_path):
                    continue
                shard_name = Path(shard_path).name
                m = re.match(r"^shard_(\d+)", shard_name)
                shard_id = int(m.group(1)) if m else None
                print(
                    f"NNUETrainIterableDataset: starting shard "
                    f"{shard_id if shard_id is not None else 'unknown'} ({shard_name})"
                )
                for feats, target in _iter_feature_records(shard_path):
                    yield torch.from_numpy(feats).float(), torch.tensor([target], dtype=torch.float32)
                    yielded += 1
                    if self.max_positions is not None and yielded >= self.max_positions:
                        return
            return

        if _is_features_file(self.data_file):
            yielded = 0
            for feats, target in _iter_feature_records(self.data_file):
                yield torch.from_numpy(feats).float(), torch.tensor([target], dtype=torch.float32)
                yielded += 1
                if self.max_positions is not None and yielded >= self.max_positions:
                    break
            return

        yielded = 0
        for line in self._line_iterator():
            try:
                record = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            board, raw_cp = _parse_record_to_board_and_raw_cp(record)
            if board is None or raw_cp is None:
                continue
            feats = self.feature_extractor.board_to_features(board)
            y = apply_cp_regression_target(
                raw_cp,
                self.cp_label_clip_min,
                self.cp_label_clip_max,
                self.cp_target_scale,
            )
            target_tensor = torch.tensor([y], dtype=torch.float32)
            yield torch.from_numpy(feats).float(), target_tensor
            yielded += 1
            if self.max_positions is not None and yielded >= self.max_positions:
                break


class NNUEValDataset(Dataset):
    """Simple cached validation dataset reading .zst once into memory."""

    def __init__(
        self,
        data_file: str,
        max_positions: Optional[int] = None,
        cp_label_clip_min: float = -2000.0,
        cp_label_clip_max: float = 2000.0,
        cp_target_scale: float = 1.0,
    ) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if _is_features_file(data_file):
            read = 0
            for feats, target in _iter_feature_records(data_file):
                self.samples.append(
                    (torch.from_numpy(feats).float(), torch.tensor([target], dtype=torch.float32))
                )
                read += 1
                if max_positions is not None and read >= max_positions:
                    break
            return

        extractor = NNUEFeatureExtractor()
        read = 0
        with open(data_file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
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
                        try:
                            record = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        board, raw_cp = _parse_record_to_board_and_raw_cp(record)
                        if board is None or raw_cp is None:
                            continue
                        feats = extractor.board_to_features(board)
                        y = apply_cp_regression_target(
                            raw_cp,
                            cp_label_clip_min,
                            cp_label_clip_max,
                            cp_target_scale,
                        )
                        self.samples.append(
                            (torch.from_numpy(feats).float(), torch.tensor([y], dtype=torch.float32))
                        )
                        read += 1
                        if max_positions is not None and read >= max_positions:
                            break
                    if max_positions is not None and read >= max_positions:
                        break
                if buffer.strip() and (max_positions is None or read < max_positions):
                    try:
                        record = json.loads(buffer.decode("utf-8"))
                        board, raw_cp = _parse_record_to_board_and_raw_cp(record)
                        if board is not None and raw_cp is not None:
                            feats = extractor.board_to_features(board)
                            y = apply_cp_regression_target(
                                raw_cp,
                                cp_label_clip_min,
                                cp_label_clip_max,
                                cp_target_scale,
                            )
                            self.samples.append(
                                (torch.from_numpy(feats).float(), torch.tensor([y], dtype=torch.float32))
                            )
                    except Exception:
                        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class NNUEChessDataset(Dataset):
    """
    LEGACY map-style dataset kept for reference. Not used for training anymore.
    Training now uses NNUETrainIterableDataset. Validation uses NNUEValDataset.
    """

    def __init__(
        self,
        data_file: str,
        max_positions: Optional[int] = None,
        is_validation: bool = False,
        use_caching: bool = False,
        cp_label_clip_min: float = -2000.0,
        cp_label_clip_max: float = 2000.0,
        cp_target_scale: float = 1.0,
    ):
        self.data_file = data_file
        self.max_positions = max_positions
        self.is_validation = is_validation
        self.use_caching = use_caching
        self.cp_label_clip_min = cp_label_clip_min
        self.cp_label_clip_max = cp_label_clip_max
        self.cp_target_scale = cp_target_scale

        self.feature_extractor = NNUEFeatureExtractor()

        if use_caching:
            self.cached_data = self._load_and_cache_data()
            self.dataset_length = len(self.cached_data)
        else:
            self.cached_data = None
            self.dataset_length = max_positions if max_positions else 1000000

        self.current_position = 0
        self.file_handle = None
        self.decompressor = None

        if not use_caching:
            self._initialize_streaming()

    def _load_and_cache_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load entire dataset into memory for caching mode."""
        print(f"Loading and caching data from {self.data_file}...")

        cached_data = []
        positions_loaded = 0

        try:
            with open(self.data_file, "rb") as f:
                decompressor = zstd.ZstdDecompressor()
                stream_reader = decompressor.stream_reader(f)

                buffer_data = b""
                chunk_size = 64 * 1024

                while True:
                    try:
                        chunk = stream_reader.read(chunk_size)
                        if not chunk:
                            break

                        buffer_data += chunk

                        while b"\n" in buffer_data:
                            line, buffer_data = buffer_data.split(b"\n", 1)

                            if not line.strip():
                                continue

                            try:
                                data_dict = json.loads(line.decode("utf-8"))
                                board, cp_eval, fen = self._parse_position_data(data_dict)

                                if board is not None and cp_eval is not None:
                                    features = self.feature_extractor.board_to_features(board)
                                    features_tensor = torch.from_numpy(features).float()
                                    y = apply_cp_regression_target(
                                        cp_eval,
                                        self.cp_label_clip_min,
                                        self.cp_label_clip_max,
                                        self.cp_target_scale,
                                    )
                                    target_tensor = torch.tensor([y], dtype=torch.float32)

                                    cached_data.append((features_tensor, target_tensor))
                                    positions_loaded += 1

                                    if self.max_positions and positions_loaded >= self.max_positions:
                                        break

                            except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                                continue

                    except Exception as e:
                        print(f"Streaming error during caching: {e}")
                        break

                if buffer_data.strip():
                    try:
                        data_dict = json.loads(buffer_data.decode("utf-8"))
                        board, cp_eval, fen = self._parse_position_data(data_dict)

                        if board is not None and cp_eval is not None:
                            features = self.feature_extractor.board_to_features(board)
                            features_tensor = torch.from_numpy(features).float()
                            y = apply_cp_regression_target(
                                cp_eval,
                                self.cp_label_clip_min,
                                self.cp_label_clip_max,
                                self.cp_target_scale,
                            )
                            target_tensor = torch.tensor([y], dtype=torch.float32)
                            cached_data.append((features_tensor, target_tensor))
                    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                        pass

        except Exception as e:
            print(f"Error loading cached data: {e}")
            return []

        print(f"Cached {len(cached_data)} positions")
        return cached_data

    def _initialize_streaming(self):
        """Initialize streaming mode."""
        try:
            self.file_handle = open(self.data_file, "rb")
            self.decompressor = zstd.ZstdDecompressor()
            self.stream_reader = self.decompressor.stream_reader(self.file_handle)
            self.buffer_data = b""
            self.current_line = 0
        except Exception as e:
            print(f"Error opening file {self.data_file}: {e}")
            raise

    def _parse_position_data(self, data: Dict[str, Any]) -> Tuple[Optional[chess.Board], Optional[float], Optional[str]]:
        """Parse a single position into White-perspective cp, including mapped mates."""
        try:
            fen = data["fen"]
            try:
                board = chess.Board(fen)
            except (ValueError, TypeError):
                return None, None, None

            parsed_board, raw_cp = _parse_record_to_board_and_raw_cp(data)
            if parsed_board is None or raw_cp is None:
                return None, None, None

            return board, float(raw_cp), fen

        except (KeyError, ValueError, chess.InvalidMoveError):
            return None, None, None

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_caching:
            if idx < len(self.cached_data):
                return self.cached_data[idx]
            raise IndexError(f"Index {idx} out of range for cached dataset")
        else:
            if not hasattr(self, "_streaming_items") or self._streaming_items is None:
                self._streaming_items = []
                self._streaming_position = 0

            while len(self._streaming_items) <= idx:
                try:
                    item = self._get_next_streaming_item()
                    self._streaming_items.append(item)
                except StopIteration:
                    if self._streaming_items:
                        return self._streaming_items[-1]
                    raise IndexError(f"Index {idx} out of range for streaming dataset")

            return self._streaming_items[idx]

    def _get_next_streaming_item(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.file_handle is None or self.stream_reader is None:
            raise StopIteration

        chunk_size = 64 * 1024

        while True:
            try:
                chunk = self.stream_reader.read(chunk_size)
                if not chunk:
                    raise StopIteration

                self.buffer_data += chunk

                while b"\n" in self.buffer_data:
                    line, self.buffer_data = self.buffer_data.split(b"\n", 1)

                    if not line.strip():
                        continue

                    try:
                        data_dict = json.loads(line.decode("utf-8"))
                        board, cp_eval, fen = self._parse_position_data(data_dict)

                        if board is not None and cp_eval is not None:
                            features = self.feature_extractor.board_to_features(board)
                            features_tensor = torch.from_numpy(features).float()
                            y = apply_cp_regression_target(
                                cp_eval,
                                self.cp_label_clip_min,
                                self.cp_label_clip_max,
                                self.cp_target_scale,
                            )
                            target_tensor = torch.tensor([y], dtype=torch.float32)

                            self.current_position += 1

                            if self.max_positions and self.current_position >= self.max_positions:
                                raise StopIteration

                            return features_tensor, target_tensor

                    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
                        continue

            except Exception as e:
                print(f"Streaming error: {e}")
                import traceback

                traceback.print_exc()
                raise StopIteration

    def __del__(self):
        if self.file_handle:
            self.file_handle.close()


class NNUEDataLoader:
    """Data loader factory for NNUE training."""

    @staticmethod
    def create_data_loaders(
        train_file: str,
        val_file: str,
        batch_size: int = 32,
        max_train_positions: Optional[int] = None,
        max_val_positions: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        cp_label_clip_min: float = -2000.0,
        cp_label_clip_max: float = 2000.0,
        cp_target_scale: float = 1.0,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset = NNUETrainIterableDataset(
            train_file,
            max_positions=max_train_positions,
            cp_label_clip_min=cp_label_clip_min,
            cp_label_clip_max=cp_label_clip_max,
            cp_target_scale=cp_target_scale,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
            drop_last=True,
        )

        val_dataset = NNUEValDataset(
            val_file,
            max_positions=max_val_positions,
            cp_label_clip_min=cp_label_clip_min,
            cp_label_clip_max=cp_label_clip_max,
            cp_target_scale=cp_target_scale,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        return train_loader, val_loader

    @staticmethod
    def inspect_targets(
        data_file: str,
        max_samples: int = 3,
        cp_label_clip_min: float = -2000.0,
        cp_label_clip_max: float = 2000.0,
        cp_target_scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Return parsed target summaries for sanity-check logging."""
        if Path(data_file).is_dir():
            shard_paths = list_nnue_feature_shard_paths(data_file)
            if not shard_paths:
                return []
            data_file = shard_paths[0]

        if _is_features_file(data_file):
            samples: List[Dict[str, Any]] = []
            for feats, target in _iter_feature_records(data_file):
                side_to_move = "white" if float(feats[782]) >= 0.5 else "black"
                samples.append({
                    "fen": None,
                    "side_to_move": side_to_move,
                    "selected_depth": None,
                    "selected_knodes": None,
                    "raw_cp": None,
                    "raw_mate": None,
                    "cp_clamped": None,
                    "regression_target": float(target),
                })
                if len(samples) >= max_samples:
                    break
            return samples

        samples: List[Dict[str, Any]] = []
        with open(data_file, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buffer = b""
                while len(samples) < max_samples:
                    chunk = reader.read(64 * 1024)
                    if not chunk:
                        break
                    buffer += chunk
                    while len(samples) < max_samples and b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line.decode("utf-8"))
                        except Exception:
                            continue
                        summary = _record_to_target_summary(
                            record,
                            clip_min=cp_label_clip_min,
                            clip_max=cp_label_clip_max,
                            cp_target_scale=cp_target_scale,
                        )
                        if summary is not None:
                            samples.append(summary)

                if len(samples) < max_samples and buffer.strip():
                    try:
                        record = json.loads(buffer.decode("utf-8"))
                        summary = _record_to_target_summary(
                            record,
                            clip_min=cp_label_clip_min,
                            clip_max=cp_label_clip_max,
                            cp_target_scale=cp_target_scale,
                        )
                        if summary is not None:
                            samples.append(summary)
                    except Exception:
                        pass

        return samples

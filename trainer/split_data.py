#!/usr/bin/env python3
"""
Split a large JSONL.zst file into training and validation sets.

This script randomly samples N lines for validation and puts the rest in training.
Uses a two-pass approach:
1. First pass: Count total lines (streaming, no data stored)
2. Generate random line numbers for validation set
3. Second pass: Stream once, write each line to appropriate output file

Memory usage is minimal (~16-24 MB for 1M validation line numbers).

Usage:
    python3 trainer/split_data.py --input trainer/localdata/lichess_db_eval.jsonl.zst \
                                   --val-size 1000000 \
                                   --val-output trainer/localdata/lichess_db_eval.val.1M.jsonl.zst \
                                   --train-output trainer/localdata/lichess_db_eval.train.jsonl.zst
"""

import argparse
import zstandard as zstd
import random
import sys
from pathlib import Path


def split_jsonl_zst(input_file: str, val_file: str, train_file: str, val_size: int, compression_level: int = 16):
    """
    Split a compressed JSONL file into training and validation sets.
    
    Uses a two-pass approach:
    1. First pass: Count total lines (streaming, no data stored)
    2. Generate random line numbers for validation set
    3. Second pass: Stream once, write each line to appropriate output file
    
    Args:
        input_file: Path to input .jsonl.zst file
        val_file: Path to output validation .jsonl.zst file
        train_file: Path to output training .jsonl.zst file
        val_size: Number of lines to randomly sample for validation
        compression_level: Zstd compression level (1-22, default 16)
    """
    # Verify input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found", file=sys.stderr)
        sys.exit(1)
    
    dctx = zstd.ZstdDecompressor()
    
    # Pass 1: Count total lines (streaming, no data stored)
    print(f"Pass 1: Counting lines in {input_file}...")
    total_lines = 0
    with open(input_file, 'rb') as f_in:
        with dctx.stream_reader(f_in) as stream_reader:
            buffer = b''
            while True:
                chunk = stream_reader.read(8192)  # Read 8KB chunks
                if not chunk:
                    # Process remaining buffer (last line may not have newline)
                    if buffer:
                        total_lines += 1
                    break
                buffer += chunk
                # Count complete lines
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    total_lines += 1
                    if total_lines % 10000000 == 0:
                        print(f"  Counted {total_lines:,} lines...", file=sys.stderr)
    
    print(f"Total lines: {total_lines:,}")
    
    # Validate val_size
    if val_size > total_lines:
        print(f"Error: val_size ({val_size:,}) exceeds total lines ({total_lines:,})", file=sys.stderr)
        sys.exit(1)
    
    # Generate random line numbers for validation set
    print(f"Generating {val_size:,} random line numbers for validation set...")
    val_line_nums = set(random.sample(range(1, total_lines + 1), val_size))
    print(f"Validation set: {len(val_line_nums):,} line numbers selected")
    
    # Pass 2: Stream once and write to appropriate files
    print(f"\nPass 2: Writing output files with compression level {compression_level}...")
    cctx_val = zstd.ZstdCompressor(level=compression_level, threads=-1)
    cctx_train = zstd.ZstdCompressor(level=compression_level, threads=-1)
    val_written = 0
    train_written = 0
    
    with open(input_file, 'rb') as f_in, \
         open(val_file, 'wb') as f_val, \
         open(train_file, 'wb') as f_train:
        
        with dctx.stream_reader(f_in) as stream_reader:
            with cctx_val.stream_writer(f_val) as val_writer, \
                 cctx_train.stream_writer(f_train) as train_writer:
                
                line_num = 0
                buffer = b''
                while True:
                    chunk = stream_reader.read(8192)  # Read 8KB chunks
                    if not chunk:
                        # Process remaining buffer (last line may not have newline)
                        if buffer:
                            line_num += 1
                            if line_num in val_line_nums:
                                val_writer.write(buffer)
                                val_written += 1
                            else:
                                train_writer.write(buffer)
                                train_written += 1
                        break
                    buffer += chunk
                    # Process complete lines
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line_num += 1
                        
                        # Write line with newline
                        line_with_newline = line + b'\n'
                        if line_num in val_line_nums:
                            val_writer.write(line_with_newline)
                            val_written += 1
                        else:
                            train_writer.write(line_with_newline)
                            train_written += 1
                        
                        if line_num % 1000000 == 0:
                            print(f"  Written {line_num:,} lines (val: {val_written:,}, train: {train_written:,})...", file=sys.stderr)
    
    print(f"\nDone!")
    print(f"Validation file: {val_file} ({val_written:,} lines)")
    print(f"Training file: {train_file} ({train_written:,} lines)")


def main():
    parser = argparse.ArgumentParser(
        description='Split a large JSONL.zst file into training and validation sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with 1M validation lines
  python3 trainer/split_data.py --input trainer/localdata/lichess_db_eval.jsonl.zst \\
                                --val-size 1000000

  # Custom output paths
  python3 trainer/split_data.py --input data.jsonl.zst \\
                                --val-size 500000 \\
                                --val-output val.jsonl.zst \\
                                --train-output train.jsonl.zst
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input .jsonl.zst file to split')
    parser.add_argument('--val-size', type=int, default=1000000,
                       help='Number of lines to randomly sample for validation (default: 1000000)')
    parser.add_argument('--val-output', type=str, default=None,
                       help='Output path for validation file (default: <input>.val.<N>.jsonl.zst)')
    parser.add_argument('--train-output', type=str, default=None,
                       help='Output path for training file (default: <input>.train.jsonl.zst)')
    parser.add_argument('--compression-level', type=int, default=16, choices=range(1, 23),
                       help='Zstd compression level 1-22 (default: 16)')
    
    args = parser.parse_args()
    
    # Generate default output paths if not provided
    input_path = Path(args.input)
    if args.val_output is None:
        args.val_output = str(input_path.parent / f"{input_path.stem.replace('.jsonl', '')}.val.{args.val_size//1000000}M.jsonl.zst")
    if args.train_output is None:
        args.train_output = str(input_path.parent / f"{input_path.stem.replace('.jsonl', '')}.train.jsonl.zst")
    
    # Run the split
    split_jsonl_zst(
        input_file=args.input,
        val_file=args.val_output,
        train_file=args.train_output,
        val_size=args.val_size,
        compression_level=args.compression_level
    )


if __name__ == "__main__":
    main()


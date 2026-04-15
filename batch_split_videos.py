#!/usr/bin/env python3
"""Batch split long videos into 2-second clips for VBench Long evaluation.

Reads a list of video folders, splits each .mp4 file in parallel into
2s clips under `<folder>/split_clip/<video_name>/*.mp4`.

Designed to run on a CPU node so GPU nodes don't waste time on ffmpeg.

Usage:
    python3 batch_split_videos.py --workers 64 \\
        /path/to/folder1 /path/to/folder2 ...
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sure vbench2_beta_long is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def split_one(video_path, split_clip_dir, duration=2):
    """Split a single video. Returns (video_path, success, msg)."""
    from vbench2_beta_long.utils import split_video_into_clips
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_dir = os.path.join(split_clip_dir, video_name)
    # Skip if already fully split (check folder exists and has mp4 files)
    if os.path.isdir(target_dir):
        existing = [f for f in os.listdir(target_dir) if f.endswith('.mp4')]
        if len(existing) > 0:
            return (video_path, True, f"skip ({len(existing)} clips exist)")
    try:
        split_video_into_clips(video_path, split_clip_dir, duration=duration)
        return (video_path, True, "done")
    except Exception as e:
        return (video_path, False, f"ERROR: {e}")


def collect_tasks(folders):
    """Collect (video_path, split_clip_dir) for all .mp4 under each folder."""
    tasks = []
    for folder in folders:
        folder = os.path.abspath(folder)
        if not os.path.isdir(folder):
            print(f"[WARN] Not a directory: {folder}")
            continue
        split_clip_dir = os.path.join(folder, "split_clip")
        os.makedirs(split_clip_dir, exist_ok=True)
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".mp4") and os.path.isfile(os.path.join(folder, fname)):
                tasks.append((os.path.join(folder, fname), split_clip_dir))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Video folders to process")
    ap.add_argument("--workers", type=int, default=64,
                    help="Number of parallel worker processes")
    ap.add_argument("--duration", type=int, default=2,
                    help="Clip duration in seconds (VBench Long uses 2)")
    args = ap.parse_args()

    start = time.time()
    tasks = collect_tasks(args.folders)
    print(f"Collected {len(tasks)} videos across {len(args.folders)} folders")
    print(f"Using {args.workers} workers, duration={args.duration}s")
    print()

    if not tasks:
        print("No videos to process, exiting.")
        return

    done = 0
    failed = 0
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(split_one, vp, sd, args.duration): vp
            for vp, sd in tasks
        }
        for future in as_completed(futures):
            video_path, ok, msg = future.result()
            done += 1
            if not ok:
                failed += 1
                print(f"[{done}/{total}] FAIL {video_path}: {msg}")
            else:
                if done % 20 == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"[{done}/{total}] {msg} "
                          f"({rate:.2f} vid/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print()
    print(f"Done: {done - failed}/{total} succeeded, {failed} failed, "
          f"elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()

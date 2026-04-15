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

from tqdm import tqdm

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
            return (video_path, True, "skip")
    try:
        split_video_into_clips(video_path, split_clip_dir, duration=duration)
        return (video_path, True, "done")
    except Exception as e:
        return (video_path, False, f"ERROR: {e}")


def collect_folder_tasks(folder):
    """Return (split_clip_dir, list of video paths) for one folder."""
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return None, []
    split_clip_dir = os.path.join(folder, "split_clip")
    os.makedirs(split_clip_dir, exist_ok=True)
    videos = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".mp4") and os.path.isfile(os.path.join(folder, fname)):
            videos.append(os.path.join(folder, fname))
    return split_clip_dir, videos


def process_folder(folder, workers, duration, folder_idx, total_folders):
    """Split all videos in one folder with a progress bar."""
    split_clip_dir, videos = collect_folder_tasks(folder)
    if not videos:
        print(f"[{folder_idx}/{total_folders}] SKIP (no videos): {folder}")
        return 0, 0, 0

    label = os.path.basename(os.path.dirname(folder)) + "/" + os.path.basename(folder)
    print(f"\n[{folder_idx}/{total_folders}] {label}  ({len(videos)} videos)", flush=True)

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(split_one, vp, split_clip_dir, duration): vp
            for vp in videos
        }
        pbar = tqdm(total=len(videos), desc=f"  split", unit="vid",
                    ncols=100, leave=True)
        for future in as_completed(futures):
            vp, ok, msg = future.result()
            if not ok:
                failed += 1
                pbar.write(f"  FAIL: {os.path.basename(vp)}: {msg}")
            elif msg == "skip":
                skipped += 1
            else:
                done += 1
            pbar.set_postfix({"done": done, "skip": skipped, "fail": failed})
            pbar.update(1)
        pbar.close()

    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Video folders to process")
    ap.add_argument("--workers", type=int, default=64,
                    help="Number of parallel worker processes")
    ap.add_argument("--duration", type=int, default=2,
                    help="Clip duration in seconds (VBench Long uses 2)")
    args = ap.parse_args()

    start = time.time()
    total_folders = len(args.folders)
    print(f"Processing {total_folders} folders with {args.workers} workers, "
          f"duration={args.duration}s")

    tot_done = tot_skip = tot_fail = 0
    for i, folder in enumerate(args.folders, start=1):
        d, s, f = process_folder(folder, args.workers, args.duration, i, total_folders)
        tot_done += d
        tot_skip += s
        tot_fail += f

    elapsed = time.time() - start
    print()
    print(f"=== ALL FOLDERS DONE ===")
    print(f"  new splits : {tot_done}")
    print(f"  skipped    : {tot_skip}")
    print(f"  failed     : {tot_fail}")
    print(f"  elapsed    : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

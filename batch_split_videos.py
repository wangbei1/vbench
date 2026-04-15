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
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def get_video_info(video_path):
    """Return (nb_frames, fps, width, height) using ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=nb_frames,r_frame_rate,duration,width,height',
             '-of', 'json', video_path],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        num, den = stream['r_frame_rate'].split('/')
        fps = float(num) / float(den)
        if 'nb_frames' in stream and stream['nb_frames'].isdigit():
            nb = int(stream['nb_frames'])
        else:
            duration = float(stream.get('duration', 0))
            nb = int(round(duration * fps))
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))
        return nb, fps, width, height
    except Exception:
        return None, None, None, None


def expected_clip_count(video_path, clip_duration):
    """Mirror the math in split_video_into_clips to compute expected clips."""
    nb_frames, fps, _, _ = get_video_info(video_path)
    if nb_frames is None or fps is None or fps <= 0:
        return None
    segment_frame_count = int(fps * clip_duration)
    if segment_frame_count <= 0:
        return None
    if nb_frames < segment_frame_count:
        return 1  # saved as single "_full" clip
    total_segments = nb_frames // segment_frame_count
    remaining = nb_frames % segment_frame_count
    return total_segments + (1 if remaining > 0 else 0)


def estimate_mem_per_video_gb(video_path):
    """Estimate peak memory (GB) needed to load one video as float32 tensor."""
    nb_frames, _, w, h = get_video_info(video_path)
    if not nb_frames or not w or not h:
        return None
    # float32 tensor + safety margin 2x for intermediate copies
    bytes_needed = nb_frames * w * h * 3 * 4 * 2
    return bytes_needed / (1024 ** 3)


def adaptive_workers(videos, max_workers, mem_budget_gb):
    """Pick worker count based on probed memory per video."""
    if not videos:
        return max_workers
    mem_per = estimate_mem_per_video_gb(videos[0])
    if mem_per is None or mem_per <= 0:
        return max_workers
    safe = max(1, int(mem_budget_gb / mem_per))
    return min(max_workers, safe)

# Make sure vbench2_beta_long is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def split_one(video_path, split_clip_dir, duration=2):
    """Split a single video. Returns (video_path, success, msg)."""
    from vbench2_beta_long.utils import split_video_into_clips
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_dir = os.path.join(split_clip_dir, video_name)

    # Skip only if the number of clips matches the expected count exactly
    if os.path.isdir(target_dir):
        existing = [f for f in os.listdir(target_dir) if f.endswith('.mp4')]
        expected = expected_clip_count(video_path, duration)
        if expected is not None and len(existing) == expected:
            return (video_path, True, "skip")
        # Incomplete split: wipe and redo
        shutil.rmtree(target_dir, ignore_errors=True)

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


def process_folder(folder, workers, duration, folder_idx, total_folders,
                   mem_budget_gb):
    """Split all videos in one folder with a progress bar."""
    split_clip_dir, videos = collect_folder_tasks(folder)
    if not videos:
        print(f"[{folder_idx}/{total_folders}] SKIP (no videos): {folder}")
        return 0, 0, 0

    label = os.path.basename(os.path.dirname(folder)) + "/" + os.path.basename(folder)
    # Adapt worker count to video memory footprint
    eff_workers = adaptive_workers(videos, workers, mem_budget_gb)
    mem_per = estimate_mem_per_video_gb(videos[0])
    mem_str = f"{mem_per:.1f}GB/vid" if mem_per else "?GB/vid"
    print(f"\n[{folder_idx}/{total_folders}] {label}  "
          f"({len(videos)} videos, {mem_str}, workers={eff_workers}/{workers})",
          flush=True)

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=eff_workers) as executor:
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
                    help="Max parallel worker processes (may be reduced per-folder)")
    ap.add_argument("--duration", type=int, default=2,
                    help="Clip duration in seconds (VBench Long uses 2)")
    ap.add_argument("--mem-budget-gb", type=float, default=300.0,
                    help="Total RAM budget for concurrent video tensors (GB)")
    args = ap.parse_args()

    start = time.time()
    total_folders = len(args.folders)
    print(f"Processing {total_folders} folders with max {args.workers} workers, "
          f"duration={args.duration}s, mem_budget={args.mem_budget_gb:.0f}GB")

    tot_done = tot_skip = tot_fail = 0
    for i, folder in enumerate(args.folders, start=1):
        d, s, f = process_folder(folder, args.workers, args.duration, i,
                                 total_folders, args.mem_budget_gb)
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

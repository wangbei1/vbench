#!/usr/bin/env python3
"""Stream filter that tees stdin to a log file and also emits progress
pings at 20% boundaries whenever it sees a tqdm-style `NN%|` progress bar.

Used by run_all_folders_eval.sh to give live intra-dimension progress
updates without losing any of the raw eval output.

Behaviour:
  - Everything read on stdin is appended verbatim to --log (splitting on
    both \\n and \\r so tqdm's carriage-return updates land as separate
    log entries).
  - When a chunk matches `\\b(\\d{1,3})%\\|`, we compute which 20% bucket
    it falls into and, if that bucket is strictly greater than the last
    one we emitted, print one line to our own stdout:
        ..{pct:>3d}%  {label}  {dim}
    Serialized across workers via flock on --lockfile so parallel
    folders never interleave.
  - The 0% and 100% buckets are suppressed (0% is meaningless, 100% is
    already covered by the DONE line emitted by the shell).
"""
import argparse
import fcntl
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--log', required=True)
ap.add_argument('--label', required=True)
ap.add_argument('--dim', required=True)
ap.add_argument('--lockfile', required=True)
args = ap.parse_args()

log_f = open(args.log, 'ab')
lock_fd = open(args.lockfile, 'w')

pct_re = re.compile(rb'\b(\d{1,3})%\|')
BUCKETS = (20, 40, 60, 80)
last_bucket = 0

buf = b''
while True:
    try:
        chunk = sys.stdin.buffer.read(512)
    except KeyboardInterrupt:
        break
    if not chunk:
        break
    buf += chunk
    # Split on \r and \n so tqdm updates (\r-terminated) are handled.
    parts = re.split(rb'[\r\n]', buf)
    buf = parts.pop()
    for p in parts:
        if not p:
            continue
        log_f.write(p + b'\n')
        log_f.flush()
        m = pct_re.search(p)
        if not m:
            continue
        pct = int(m.group(1))
        # Pick the highest bucket whose boundary we have crossed.
        bucket = 0
        for b in BUCKETS:
            if pct >= b:
                bucket = b
        if bucket > last_bucket:
            last_bucket = bucket
            line = f'  ..{bucket:>3d}%  {args.label}  {args.dim}\n'
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                sys.stdout.write(line)
                sys.stdout.flush()
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

if buf:
    log_f.write(buf + b'\n')
log_f.close()

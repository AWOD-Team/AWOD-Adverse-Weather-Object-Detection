#!/usr/bin/env python
"""Upload the AWOD project code/weights to the training server over SFTP.

Connection parameters come from the same environment variables as
tools/ssh_remote.py (AWOD_SSH_HOST / AWOD_SSH_PORT / AWOD_SSH_USER /
AWOD_SSH_PASSWORD).

Usage:
    python tools/sync_to_server.py                # code + small weights
    python tools/sync_to_server.py --weights all   # also yolo11l/m
    python tools/sync_to_server.py --dataset       # also push dataset_built
    python tools/sync_to_server.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import paramiko

HOST = os.environ.get("AWOD_SSH_HOST", "175.155.64.171")
PORT = int(os.environ.get("AWOD_SSH_PORT", "24068"))
USER = os.environ.get("AWOD_SSH_USER", "linux")
PASSWORD = os.environ.get("AWOD_SSH_PASSWORD", "")

REPO = Path(__file__).resolve().parent.parent
REMOTE_ROOT = os.environ.get("AWOD_REMOTE_ROOT", "/data/awod")

# directories copied verbatim (relative to the repo root)
CODE_DIRS = ["core", "demo", "yolo_baseline", "tools", "scripts"]
CODE_FILES = [
    "README.md",
    "requirements.txt",
    "第一次任务计划.md",
    "第二次任务计划.md",
    # "正式任务计划" is intentionally NOT uploaded: it contains the server
    # password in plaintext and the server does not need a copy of it.
    "train/train_foggy.py",
    "train/train_rtts.py",
    "train/requirements.txt",
]
SMALL_WEIGHTS = ["yolo11n.pt", "yolo11s.pt", "yolo26n.pt"]
LARGE_WEIGHTS = ["yolo11l.pt", "yolo11m.pt", "train/yolo11l.pt", "train/yolo11m.pt"]

SKIP_DIRS = {".git", ".venv", ".idea", "__pycache__", "runs", "dehaze_result",
             "weighs", "dataset_rtts", "dataset_foggy", "awod_cuda"}
SKIP_SUFFIX = {".pyc", ".zip", ".lnk"}

DATASET_LOCAL = Path("G:/leetcode/dataset_built")

# (image dir, label dir) pairs that make up the real fog dataset sent to the
# server as <remote>/dataset/{images,labels}
FOG_PAIRS = [("images_1", "annonation/labels_1"),
             ("images_2", "annonation/labels_2")]


def iter_local_files(paths, repo: Path, skip_dirs, skip_suffix):
    """Yield (local_path, repo_relative_posix_path) for regular files."""
    for rel in paths:
        local = (repo / rel).resolve()
        if local.is_file():
            if local.suffix not in skip_suffix:
                yield local, rel
        elif local.is_dir():
            for root, dirs, files in os.walk(local):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                for name in files:
                    p = Path(root) / name
                    if p.suffix in skip_suffix:
                        continue
                    r = p.resolve().relative_to(repo).as_posix()
                    yield p, r
        else:
            print(f"[warn] missing locally, skipped: {local}")


def sftp_makedirs(sftp, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", choices=["none", "small", "all"], default="small")
    ap.add_argument("--dataset", action="store_true",
                    help="also upload G:/leetcode/dataset_built")
    ap.add_argument("--dataset-smoke", type=int, metavar="N", default=0,
                    help="upload only N images (+labels) per source, for a "
                         "quick pipeline smoke test")
    ap.add_argument("--fog-ds", action="store_true",
                    help="upload images_1/images_2 + labels as "
                         "<remote>/dataset/{images,labels} (flat YOLO layout)")
    ap.add_argument("--rtts", action="store_true",
                    help="upload dataset_built/RTTS (real fog, VOC xml) as "
                         "<remote>/rtts/{JPEGImages,Annotations}")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=8,
                    help="并行 SFTP 连接数（默认 8）。大量小文件时串行会被"
                         "往返延迟拖慢，并行可显著加速")
    args = ap.parse_args()

    if not PASSWORD:
        print("[ssh] AWOD_SSH_PASSWORD is not set", file=sys.stderr)
        return 2

    targets = list(CODE_DIRS) + list(CODE_FILES)
    if args.weights == "small":
        targets += SMALL_WEIGHTS
    elif args.weights == "all":
        targets += SMALL_WEIGHTS + LARGE_WEIGHTS

    plan = list(iter_local_files(targets, REPO, SKIP_DIRS, SKIP_SUFFIX))
    if args.fog_ds:
        if not DATASET_LOCAL.is_dir():
            print(f"[ssh] dataset root not found: {DATASET_LOCAL}", file=sys.stderr)
            return 2
        seen: dict[str, str] = {}
        for img_dir, lbl_dir in FOG_PAIRS:
            for p in sorted((DATASET_LOCAL / img_dir).iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if p.name in seen:
                    print(f"[ssh] duplicate image name across sources: {p.name}"
                          f" (already from {seen[p.name]})", file=sys.stderr)
                    return 2
                seen[p.name] = img_dir
                plan.append((p, f"dataset/images/{p.name}"))
                lp = DATASET_LOCAL / lbl_dir / f"{p.stem}.txt"
                if lp.is_file():
                    plan.append((lp, f"dataset/labels/{lp.name}"))
                else:
                    print(f"[warn] no label for {p.name}")
        print(f"[fog-ds] images={len(seen)}")
    if args.rtts:
        src = DATASET_LOCAL / "RTTS"
        if not src.is_dir():
            print(f"[ssh] RTTS not found: {src}", file=sys.stderr)
            return 2
        for root_dir, dirs, files in os.walk(src):
            for name in files:
                p = Path(root_dir) / name
                if p.suffix == ".pyc":
                    continue
                rel = p.resolve().relative_to(src.resolve()).as_posix()
                plan.append((p, f"rtts/{rel}"))
    if args.dataset_smoke:
        if not DATASET_LOCAL.is_dir():
            print(f"[ssh] dataset root not found: {DATASET_LOCAL}", file=sys.stderr)
            return 2
        for img_dir, lbl_dir in [("images_1", "annonation/labels_1"),
                                 ("images_2", "annonation/labels_2")]:
            imgs = sorted(p for p in (DATASET_LOCAL / img_dir).iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            for p in imgs[:args.dataset_smoke]:
                plan.append((p, f"datasets/{img_dir}/{p.name}"))
                lp = DATASET_LOCAL / lbl_dir / f"{p.stem}.txt"
                if lp.is_file():
                    plan.append((lp, f"datasets/{lbl_dir}/{lp.name}"))
                else:
                    print(f"[warn] no label for {p.name}")
    if args.dataset:
        if not DATASET_LOCAL.is_dir():
            print(f"[ssh] dataset root not found: {DATASET_LOCAL}", file=sys.stderr)
            return 2
        for root, dirs, files in os.walk(DATASET_LOCAL):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS
                       and d != "venv_labelimg"]
            for name in files:
                p = Path(root) / name
                if p.suffix == ".pyc":
                    continue
                r = "datasets/" + p.resolve().relative_to(
                    DATASET_LOCAL.resolve()).as_posix()
                plan.append((p, r))

    total = sum(p.stat().st_size for p, _ in plan)
    print(f"files={len(plan)}  total={total / 1024 / 1024:.1f} MB  "
          f"-> {REMOTE_ROOT}")
    if args.dry_run:
        for _, r in sorted(plan, key=lambda x: x[1]):
            print("  ", r)
        return 0

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD,
                   timeout=20, banner_timeout=30, auth_timeout=30,
                   look_for_keys=False, allow_agent=False)

    plan.sort(key=lambda x: x[1])
    tls = threading.local()
    lock = threading.Lock()
    state = {"done": 0, "uploaded": 0, "failed": 0}

    def thread_sftp() -> paramiko.SFTPClient:
        s = getattr(tls, "sftp", None)
        if s is None:
            s = client.open_sftp()
            tls.sftp = s
        return s

    def drop_thread_sftp() -> None:
        s = getattr(tls, "sftp", None)
        if s is not None:
            try:
                s.close()
            except Exception:
                pass
            tls.sftp = None

    def work(item) -> None:
        local, rel = item
        remote = posixpath.join(REMOTE_ROOT, rel)
        size = local.stat().st_size
        for attempt in (1, 2, 3):
            try:
                sftp = thread_sftp()
                try:
                    # 已传完（大小一致）则跳过 —— 这就是断点续传的判据
                    if sftp.stat(remote).st_size == size:
                        with lock:
                            state["done"] += 1
                        return
                except IOError:
                    pass
                sftp.put(str(local), remote)
                with lock:
                    state["done"] += 1
                    state["uploaded"] += 1
                    n = state["done"]
                if n % 50 == 0 or n == len(plan):
                    print(f"  [{n}/{len(plan)}] uploaded={state['uploaded']}"
                          f" failed={state['failed']}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - retry with a fresh channel
                drop_thread_sftp()
                if attempt == 3:
                    with lock:
                        state["done"] += 1
                        state["failed"] += 1
                    print(f"  [fail] {rel}: {exc}", file=sys.stderr, flush=True)

    try:
        # 目录先串行建好，避免多线程竞争
        setup = client.open_sftp()
        made: set[str] = set()
        for _, rel in plan:
            rdir = posixpath.dirname(posixpath.join(REMOTE_ROOT, rel))
            if rdir and rdir not in made:
                sftp_makedirs(setup, rdir)
                made.add(rdir)
        setup.close()

        jobs = max(1, args.jobs)
        if jobs == 1:
            for item in plan:
                work(item)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                for _ in pool.map(work, plan):
                    pass
    finally:
        drop_thread_sftp()
        client.close()

    print(f"done: {state['done']} files processed, "
          f"{state['uploaded']} transferred, {state['failed']} failed")
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

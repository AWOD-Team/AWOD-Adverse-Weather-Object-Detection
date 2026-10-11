#!/usr/bin/env python
"""Reusable non-interactive SSH helper for the AWOD training server.

Connection parameters come from environment variables (nothing secret is
hard-coded here):

    AWOD_SSH_HOST      default 175.155.64.171
    AWOD_SSH_PORT      default 24068
    AWOD_SSH_USER      default linux
    AWOD_SSH_PASSWORD  required (no default)

Usage:
    python tools/ssh_remote.py "nvidia-smi; nproc; free -g"
    python tools/ssh_remote.py --timeout 120 "bash train.sh"

Exit code is 0 when the remote command exits 0, otherwise the remote exit code
(or 2 for connection/auth failure).
"""

from __future__ import annotations

import argparse
import os
import sys

import paramiko

HOST = os.environ.get("AWOD_SSH_HOST", "175.155.64.171")
PORT = int(os.environ.get("AWOD_SSH_PORT", "24068"))
USER = os.environ.get("AWOD_SSH_USER", "linux")
PASSWORD = os.environ.get("AWOD_SSH_PASSWORD", "")


def run(command: str, timeout: float, host: str, port: int, user: str,
        password: str) -> int:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=20,
            banner_timeout=30,
            auth_timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception as exc:  # noqa: BLE001 - surface any connect failure
        print(f"[ssh] connection failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    try:
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
    finally:
        client.close()

    if out:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if err:
        sys.stderr.write(err if err.endswith("\n") else err + "\n")
    return code


def main() -> int:
    # the local Windows console is often GBK; remote output is UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", help="remote shell command to execute")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="seconds to wait for the remote command (default 60)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--user", default=USER)
    args = ap.parse_args()

    if not PASSWORD:
        print("[ssh] AWOD_SSH_PASSWORD is not set", file=sys.stderr)
        return 2
    return run(args.command, args.timeout, args.host, args.port, args.user,
               PASSWORD)


if __name__ == "__main__":
    raise SystemExit(main())

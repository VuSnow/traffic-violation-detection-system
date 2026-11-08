#!/usr/bin/env bash
# Work on the GPU server from this machine.
#   scripts/server.sh sync             make the server's src/, tests/, configs/, scripts/, docker/, pyproject.toml and .dockerignore the same as here
#   scripts/server.sh test [args]      run pytest there, e.g. scripts/server.sh test tests/tvd -q -k "yolo and notebooks"
#   scripts/server.sh run <command>    run a command there, in the project folder, with the tvd environment first on PATH
set -euo pipefail
HOST="${TVD_SERVER:-20.0.14.182}"
REMOTE_DIR="${TVD_REMOTE_DIR:-workspace/dungvu/traffic-violation-detection-system}"
REMOTE_ENV="${TVD_REMOTE_ENV:-workspace/dungvu/envs/tvd}"
SSH=(ssh -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=3)

remote() {
    "${SSH[@]}" "$HOST" "cd ~/$REMOTE_DIR && export PATH=~/$REMOTE_ENV/bin:\$PATH && $*"
}

case "${1:-}" in
    sync)
        # --delete removes files deleted or renamed here, so the server never runs a test that no longer exists.
        # Excluded files are neither copied nor deleted: each machine keeps its own caches and install records.
        rsync -az --delete --timeout=60 -e "${SSH[*]}" --exclude '__pycache__/' --exclude '*.egg-info/' \
            src tests configs scripts docker pyproject.toml .dockerignore "$HOST:$REMOTE_DIR/"
        echo "synced to $HOST:$REMOTE_DIR"
        ;;
    test)
        shift
        # printf %q quotes each argument again, so that one with spaces reaches pytest as one argument.
        args=""
        if [ $# -gt 0 ]; then args=$(printf '%q ' "$@"); fi
        remote python -m pytest "$args"
        ;;
    run)
        shift
        remote "$@"
        ;;
    *)
        echo "usage: scripts/server.sh sync | test [pytest args] | run <command>" >&2
        exit 2
        ;;
esac

#!/usr/bin/env bash
# Work on the GPU server from this machine.
#   scripts/server.sh sync             copy src/, tests/, configs/, scripts/ and pyproject.toml to the server
#   scripts/server.sh test [args]      run pytest there, e.g. scripts/server.sh test tests/tvd -q
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
        rsync -az --timeout=60 -e "${SSH[*]}" --exclude '__pycache__/' src tests configs scripts pyproject.toml "$HOST:$REMOTE_DIR/"
        echo "synced to $HOST:$REMOTE_DIR"
        ;;
    test)
        shift
        remote python -m pytest "$@"
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

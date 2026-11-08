#!/usr/bin/env bash
# The review service in Docker, on the machine this runs on (ai_server_2). Run it in the project folder there:
#   scripts/server.sh run scripts/deploy.sh <command>
#
#   build <commit>   build the image tvd-service:<commit> from docker/Dockerfile, and tag it tvd-service:latest
#   start            run tvd-service:latest: on the GPU, on port 8040 of this machine only, with its jobs in
#                    outputs/service, and started again by Docker after a crash or a reboot
#   stop             stop it and remove the container; the jobs stay in outputs/service
#   status           the container, its image and its health
#   logs [lines]     the last lines of its log (50 by default)
#   health           what GET /health says
#
# With TVD_DRY_RUN set, each docker command is printed instead of run.
set -euo pipefail
NAME=tvd-service
IMAGE=tvd-service
STORAGE="$PWD/outputs/service"

run() {
    if [ -n "${TVD_DRY_RUN:-}" ]; then echo "$*"; else "$@"; fi
}

case "${1:-}" in
    build)
        commit="${2:?the commit to name the image after, such as $(git rev-parse --short HEAD 2>/dev/null || echo 6b1a1ba)}"
        run docker build -f docker/Dockerfile --build-arg "TVD_COMMIT=$commit" -t "$IMAGE:$commit" -t "$IMAGE:latest" .
        ;;
    start)
        # The jobs' folder belongs to this user, as does the container's user (1000): both can read and write it.
        run mkdir -p "$STORAGE"
        run docker run -d --name "$NAME" --gpus all --restart unless-stopped -p 127.0.0.1:8040:8040 -v "$STORAGE:/data/service" "$IMAGE:latest"
        ;;
    stop)
        run docker stop "$NAME"
        run docker rm "$NAME"
        ;;
    status)
        run docker ps -a --filter "name=^$NAME\$" --format "{{.Names}} {{.Image}} {{.Status}}"
        ;;
    logs)
        run docker logs --tail "${2:-50}" "$NAME"
        ;;
    health)
        run curl -s localhost:8040/health
        ;;
    *)
        echo "usage: scripts/deploy.sh build <commit> | start | stop | status | logs [lines] | health" >&2
        exit 2
        ;;
esac

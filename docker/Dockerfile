# The review service, with the versions its tests passed with on ai_server_2. Build it from the project folder:
#   scripts/deploy.sh build <commit>
FROM python:3.11.18-slim-bookworm

# The full OpenCV package, which Ultralytics needs, links to these two libraries.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/app/home YOLO_CONFIG_DIR=/app/home/ultralytics
WORKDIR /app

# The libraries first, in a layer of their own: they change far less often than the code, so a new build reuses it.
# PyTorch's CUDA 13.2 build comes from PyTorch's own index. BoxMOT is installed without its dependencies, as on the
# server, because it pins an older OpenCV (see pyproject.toml).
COPY docker/constraints.txt docker/constraints.txt
COPY pyproject.toml README.md ./
RUN mkdir -p src/tvd src/traffic_violation_detection \
 && touch src/tvd/__init__.py src/traffic_violation_detection/__init__.py \
 && pip install --extra-index-url https://download.pytorch.org/whl/cu132 -c docker/constraints.txt ".[detection,tracking,plates,service]" \
 && pip install --no-deps boxmot==26.0.0

# ONNX Runtime's CUDA provider needs cuRAND. On the server it finds the one of the system's CUDA 12.4; the image has no
# system CUDA, so it gets the cuRAND of CUDA 13.2 from pip, as PyTorch gets its own CUDA libraries.
RUN pip install --no-deps nvidia-curand==10.4.2.66

# The models the tests used, inside the image: it runs the same models wherever it runs, and never downloads them.
# scripts/deploy.sh build copies the plate models from the server's cache to docker/models first.
COPY weights/yolo26m.pt weights/yolo26m.pt
COPY configs configs
COPY docker/models /app/home/.cache
COPY docker/check_plate_models.py docker/check_plate_models.py
RUN python docker/check_plate_models.py

# The code last: a change to it rebuilds only this layer.
COPY src src
RUN pip install --no-deps . && chown -R 1000:1000 /app/home

# The commit the image was built from: every record and the health check name it.
ARG TVD_COMMIT
ENV TVD_COMMIT=${TVD_COMMIT}

# The user that owns the job folders on the server, so that both can read and write them.
USER 1000:1000
EXPOSE 8040
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8040/health', timeout=8)"]
CMD ["python", "-m", "tvd.cli", "serve", "--config", "configs/service.docker.yaml"]

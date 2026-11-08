"""The REST layer (spec, Section 8). It only translates between HTTP and the service's core.

| Method | Path | Answer |
|---|---|---|
| POST | /jobs | 202 and the job; 413, 422 or 503 with the reason, and nothing is queued |
| GET | /jobs/{job_id} | the job's state, stage, progress and failure |
| GET | /jobs/{job_id}/records | records.json once the job is done; 409 before |
| GET | /jobs/{job_id}/files/{path} | one file of the job's folder |
| GET | /cameras | the configured cameras and their rules |
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from tvd.service.core import NotFound, NotReady, ReviewService
from tvd.service.intake import Refused
from tvd.service.worker import Worker


def create_app(service: ReviewService, run_worker: bool = True) -> FastAPI:
    """The API of one service. With ``run_worker``, its jobs run in the background while the app runs."""
    worker = Worker(service) if run_worker else None
    limit = service.settings.storage.max_upload_mb * 2**20

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if worker is not None:
            worker.start()
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()  # The running job ends first.
            service.close()

    app = FastAPI(title="Traffic violation review", version="1", lifespan=lifespan)

    @app.middleware("http")
    async def refuse_large_uploads(request: Request, call_next):
        """A body announced larger than the limit is refused before it is read. Without a length, the upload's own check refuses it."""
        length = request.headers.get("content-length", "")
        if request.method == "POST" and request.url.path == "/jobs" and length.isdigit() and int(length) > limit:
            return JSONResponse(status_code=413, content={"detail": f"The upload is larger than {limit / 2**20:.0f} MB"})
        return await call_next(request)

    @app.exception_handler(Refused)
    async def refused(request: Request, error: Refused) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"detail": error.reason})

    @app.exception_handler(NotFound)
    async def not_found(request: Request, error: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.post("/jobs", status_code=202)
    def submit(video: UploadFile = File(...), camera_id: str = Form(...), recorded_at: str = Form(...)) -> dict[str, Any]:
        job = service.submit(video.file, video.filename or "", camera_id, recorded_at)
        return {**job.as_json(), "status_url": f"/jobs/{job.job_id}", "records_url": f"/jobs/{job.job_id}/records"}

    @app.get("/jobs/{job_id}")
    def status(job_id: str) -> dict[str, Any]:
        return service.status(job_id).as_json()

    @app.get("/jobs/{job_id}/records")
    def records(job_id: str) -> dict[str, Any]:
        try:
            return service.records(job_id)
        except NotReady as state:
            raise HTTPException(status_code=409, detail=f"Job {job_id} is {state}: its records come when it is done") from None

    @app.get("/jobs/{job_id}/files/{path:path}")
    def file(job_id: str, path: str) -> FileResponse:
        return FileResponse(service.file(job_id, path))

    @app.get("/cameras")
    def cameras() -> list[dict[str, Any]]:
        return service.describe_cameras()

    return app

# Violation Review Service: Design

Date: 2026-11-07. Status: approved by the user, with the build decisions of Section 12.

## 1. Intent

A demo service for an academic project. A Python backend uploads a pre-recorded video from a fixed traffic camera. The service finds red-light and wrong-way candidates with the Phase 1 baseline, and returns one record per candidate, with its evidence, for a person to review on the backend.

The service never confirms a violation. Every record it returns is `pending`. Reviewing, confirming and storing are the backend's job.

## 2. Goals, non-goals, anti-goals

**Goals**

- Accept an uploaded video, with the camera it came from and a timestamp, and process it in the background.
- Return, for each candidate: a clip, three evidence images, a plate image when a plate was found, and a JSON record.
- Configure every camera, and every rule and pipeline setting, in files. Nothing that varies between cameras or that a user might tune is hard-coded.
- Deliver results within 2 hours of upload.

**Non-goals for this version**

- Live camera streams. They come after the upload flow is stable.
- Real fines, legal-grade evidence, authentication, or a production database.
- The review interface and the database of confirmed violations. The user builds them in the backend.
- Choosing between REST and a message queue. The core is independent of transport, and a thin REST layer comes first.
- Detecting stop lines automatically. A line-detection model may later propose camera configurations.

**Anti-goals: these would make it a failure, even if it runs**

- A record that looks confirmed, or a plate text shown as certain when it is only `uncertain`.
- A setting changed in a file without the record showing which settings produced it.
- One upload's failure, such as plate reading crashing, losing the violations already found.

## 3. Decided by the user

| Topic | Decision |
|---|---|
| Purpose | Demo and academic project only. The backend and its database are demos too. |
| Input | Upload of pre-recorded files first. Streams come later. |
| Backend | Python, written by the user. REST or a queue is to be decided with the backend. |
| Record schema | None on the backend yet. The service's schema is the starting point. |
| Evidence storage | A shared folder on the server, temporary. |
| Violations | Red light and wrong way. |
| Cameras | Fixed, as in the test clips. Each camera's geometry is configured in a file, never in code. |
| Rule parameters | Configurable, for example the 2 s window of the wrong-way rule. |
| Evidence images | Red-light images show the stop line and the signal state at that moment. |
| Deadline | Results within 2 hours of upload. |
| Time | The backend sends a timestamp with each upload, which is taken as the time of the video's first frame. |

## 4. Configuration

All settings live in files under `configs/`, in YAML so that each setting can carry a comment. They are validated when the service starts and again when a job starts. Unknown keys, missing keys, wrong types and values out of range are refused, with a message that names the file and the key.

**`configs/service.yaml`**: one file for the whole service.

- `storage`: the shared folder, and how many days a job's files are kept.
- `jobs`: the longest video accepted, in minutes, and the deadline. The longest video defaults to what measured speed allows within the deadline (Section 7).
- `pipeline`: detector (weights, image size, confidence), tracker, camera-motion settings, and whether to write a full annotated video (off by default: Section 7).
- `rules.red_light` and `rules.wrong_way`: every parameter now in `RULE` of `red_light.py` and `wrong_way.py`, for example `window_seconds: 2.0`.
- `plates`: every parameter now in `PLATE_SETTINGS`.
- `evidence`: clip length before and after the trigger, time between evidence images (now 0.7 s), image quality.

**`configs/cameras/<camera_id>.yaml`**: one file per camera.

- The frame size the camera produces, which every video must match.
- Whether the camera moves. Only a moving camera runs the camera-motion step, which was 90% of the frame time on a moving camera in Notebook 08. The default is a fixed camera.
- The rules that apply to it, one or both.
- For red light: the stop line, lane dividers, lane direction, traffic direction, and the position of each signal head with how to read it.
- For wrong way: the directional regions, with their allowed directions.
- Optional overrides of any rule parameter, for this camera only.

**Defaults.** The defaults are the values validated in Notebooks 05, 05b and 06. Red light defaults to rule version 2, the pipeline's current default. Changing a value makes the scores of Notebooks 05b and 09 no longer apply, so every record stores the effective settings that produced it (Section 6) and the version of its configuration files.

The existing scene files in `configs/scenes/` describe test videos, and red-light scenes have frame ranges (clips) because `rlr_occasions.mp4` is a compilation. A camera file describes one view for the whole video. A converter or loader adapter keeps the notebooks working.

## 5. Processing

1. **Upload.** The video is saved in the job's folder and kept with the job's other files until the retention ends. The service checks the camera exists, the file is a readable video of that camera's frame size, and it is not longer than the configured limit. If anything fails, the upload is refused with a reason, and nothing is queued.
2. **Queue.** Jobs run one at a time on the GPU, in order of arrival. A job is `queued`, `running`, `done` or `failed`. Its state is kept in a small SQLite table, so that a restart can report it. A job that was running during a restart is marked `failed`, with the reason.
3. **One pass over the video.** Detection and tracking run once, and each rule enabled for the camera is applied to the same tracks. Today each rule's function runs its own detection and tracking. Plate models load once, when the service starts, and stay in their worker process.
4. **Evidence.** For each candidate: a clip with original pixels, three images and the plate image, as in Notebook 07. Red-light images also show the stop line and the signal state that the rule read at that frame.
5. **Records.** Built as in Notebook 07, with duplicates merged, all `pending`.
6. **Failures.** A failed plate step keeps the violations, with `plate_not_attempted`, as `run_pipeline` does now. A failure that leaves nothing to return marks the job `failed` with the reason.

## 6. Output

Each job has a folder in the shared storage: `<storage>/<job_id>/` with `records.json`, `clips/`, `evidence/` and `plates/`.

**A record** follows `docs/event_schema.md` as version 2, which adds:

- `camera_id`, and the `recorded_at` timestamp sent by the backend;
- absolute times: `recorded_at` plus the seconds into the video, in ISO 8601 with the time zone sent;
- `settings`: the effective rule, plate and evidence settings, with the configuration's version;
- paths to the clip and images relative to the job's folder, which the backend can read from the shared folder or fetch over HTTP.

## 7. Speed and the 2-hour deadline

Measured on ai_server_2 (Notebook 09): the fixed-camera pipeline ran at 0.44× real time, which allows about 50 minutes of video in 2 hours. Detection and tracking alone run at 46 frames per second on dense 1080p traffic. Most of the rest is CPU work: writing the full annotated video (83 of 394 s on real footage), plus the annotated video and clips written in the first step.

The service writes only each candidate's clip by default. The longest accepted video is then set from a measurement on this server, with a margin, rather than assumed.

## 8. Interface

The core is a Python API: submit a job, get its status, get its records, list cameras. A thin REST layer, with FastAPI and uvicorn, exposes it first:

| Method | Path | Does |
|---|---|---|
| `POST` | `/jobs` | Upload a video with `camera_id` and `recorded_at`. Returns `202` and a job id, or `4xx` with a reason. |
| `GET` | `/jobs/{id}` | The job's state, its progress, and its failure reason if it failed. |
| `GET` | `/jobs/{id}/records` | The records, once the job is `done`. |
| `GET` | `/jobs/{id}/files/{path}` | One evidence file, for a backend that cannot read the shared folder. |
| `GET` | `/cameras` | The configured cameras and their rules. |

The port is configurable. Port 8000 is already taken by another service on ai_server_2.

## 9. Testing

- Unit tests for configuration loading and validation, the job queue, the shared pass, the drawing on red-light images, and absolute times, with the fake detector and tracker of the existing tests.
- API tests with FastAPI's test client.
- One end-to-end run on the server, on a short clip, before handing over.
- The service must reproduce the records of Notebook 08 for one clip of `rlr_occasions.mp4`, cut into its own file with a camera file of its own, with default settings.

## 10. Calls made by the assistant

- Configuration in YAML rather than JSON, for comments. It adds PyYAML (MIT).
- FastAPI and uvicorn for the REST layer (MIT and BSD).
- One job at a time, with job state in SQLite.
- One detection and tracking pass per video, shared by both rules.
- No full annotated video by default.
- `recorded_at` is the time of the video's first frame.
- The uploaded video is kept in the job's folder until the retention ends.
- Evidence served both from the shared folder and over HTTP.
- Deployment on ai_server_2, in `~/workspace/dungvu`.

## 11. Left to the builder

- The module layout of the service, and how the rule functions are split so that both rules use one pass.
- The exact YAML layout, within Section 4.
- How progress is measured, for example frames processed.
- How old job folders are cleaned up, within the configured retention.

## 12. How it is built (decided by the user, 2026-11-07)

- **Each part has a step-by-step plan** in `docs/superpowers/plans/`, so that the user can build it, or build it again, to learn how such a system is made, from the algorithms to deployment: one unit per step, with its full code, the command to run and the output to expect. The plans of Parts 1 to 4 were run on ai_server_2 before they were handed over, and the assistant then carried out each from its plan, at the user's request. Part 5 was built first, at the user's request, and its plan written from its commits.
- **The algorithms are rewritten** in a new package, `src/tvd/`. The notebooks' package, `traffic_violation_detection`, stays as it is, and it is the reference: wherever the new code is compared with it, the results must match exactly.
- **SOLID, with replaceable components.** Each kind of component (detector, tracker, rule, plate reader, storage) has an abstract base class and a registry. A settings file names the implementation, a factory builds it, and the pipeline only knows the base classes. Changing ByteTrack for OC-SORT is a change of one word in a YAML file.
- **Five parts,** each ending with working, tested software:

| Part | Builds |
|---|---|
| 1. Foundations and perception | Data types, registry, video reader, detector, trackers, settings, a `track` command |
| 2. Scene and rules | Camera files, rule settings, camera motion, signal reading, the wrong-way and red-light rules, and one pass that applies a camera's rules (`check`) |
| 3. Plates and evidence | Plate detection and OCR in a worker process, voting, clips, evidence images, records |
| 4. The service | Job queue, storage, REST API, around the one pass of Part 2 |
| 5. Deployment and MLOps | Docker, running on ai_server_2, logging, health checks, model versions |


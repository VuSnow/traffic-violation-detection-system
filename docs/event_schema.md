# Violation Event Schema

The record of one candidate violation, from Notebook 07. Code: `src/traffic_violation_detection/events.py`.

A record is a **candidate**. It starts `pending`, and only a reviewer's decision changes that. Nothing in the pipeline confirms a violation by itself.

## Version 2: the review service

The review service (`src/tvd/records.py`) writes version 2. It keeps every field of version 1 below, with the same meaning, and changes these:

| Field | Version 2 |
| --- | --- |
| `schema_version` | 2 |
| `event_id` | `evt-<video_id>-<rule>-f<trigger frame>-t<track id>`, for both rules |
| `camera_id` | New: the camera file the video was checked with |
| `video` | Adds `recorded_at`: the time of the video's first frame, as the backend sent it, in ISO 8601 with its time zone |
| `track` | `tracking_session` is `<video_id>/<tracker>`. There is no `clip`: a camera file describes the whole video |
| `time` | Adds `start_at`, `trigger_at` and `end_at`: `recorded_at` plus the seconds into the video, to the millisecond, in the same time zone. Red-light records now have a start and an end too: the first and last frames of the vehicle's track |
| `rule` | Only `name` and `version`. All the rule's settings are in `settings.rule` |
| `evidence` | Paths are relative to the folder of `records.json`: `clips/`, `evidence/` and `plates/`. Adds `clip_frames`: the clip's first frame and one past its last, in the video |
| `settings` | New: the effective settings that produced the record, as `perception`, `rule`, `plates` and `evidence` |
| `config_files` | New: the version of each settings file, as `sha256:` and the first 12 hex digits of its hash |
| `models` | New: the models and software behind the record. `detector` names its weights, their version and its device (`cuda:0` or `cpu`); `tracker` its type and options; `plates` the plate models and their ONNX providers, the first being `CUDAExecutionProvider` on the GPU, or `unavailable` with the reason; `software` the versions of the libraries; `code` the commit of the Docker image, or null |
| `review` | Always `pending`, with no reviewer: the service never decides. Review happens in the backend |

The file is `{"schema_version": 2, "events": [...]}`.

## Candidate record


| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | int | Version of this schema, now 1 |
| `event_id` | string | Unique, names its video, and never equals a track id |
| `video` | object | `video_id`, `fps`, `width`, `height` |
| `violation_type` | string | `red_light` or `wrong_way` |
| `track` | object | `tracking_session` (scopes the track to one video), `clip`, `track_id`, `class_name` |
| `time` | object | `trigger_frame` and `trigger_time_s` are required; `start_*` and `end_*` are null when a rule has a single moment |
| `rule` | object | `name`, `version`, and a rule's own settings |
| `why` | string | The reason, in a sentence a reviewer can check |
| `signal_source` | string or null | `camera`, or `simulated_overlay` for a signal drawn over the video; null for rules with no signal |
| `measurements` | object | What the rule measured, such as the lane, the signal state, and whether the entry was seen |
| `evidence` | object | `clip`, `plate_crop` and `frames`, as paths relative to the file of records. Frames are `before`, `entry` and `after`, or `estimated_entry`, `first_seen` and `after` for a vehicle hidden at its entry |
| `license_plate` | object | `status`, `text`, `votes`, `usable_reads`, `agreement`, `mean_prob` |
| `needs_attention` | list of strings | What the record alone cannot settle |
| `review` | object | `status`, `reviewer`, `decided_at`, `corrected_plate`, `notes` |

Times are seconds from the start of the video. Boxes are `xyxy` pixels of the frame, with the origin at the top left. A real case would also need the camera's identity, its location and the date and time, which recorded test videos do not provide.

### Plate status

| `status` | `text` | Meaning |
| --- | --- | --- |
| `read` | the plate | Enough frames agree |
| `uncertain` | a suggestion | Too few frames agree, so this is only a hint |
| `unreadable` | null | A plate was found, but no frame gave a usable text |
| `no_plate_found` | null | No plate was found on this vehicle |
| `not_attempted` | null | Plate reading did not run for this event |

### Attention flags

| Flag | The reviewer should check |
| --- | --- |
| `simulated_signal` | The signal is a graphic added to the video |
| `entry_estimated` | The crossing was hidden, and its time was estimated from how the vehicle moved afterwards |
| `entry_not_seen` | The vehicle's track starts after its front had crossed |
| `plate_uncertain` | The plate text is only a suggestion |
| `plate_unreadable`, `plate_not_found`, `plate_not_attempted` | No plate text is available |

## Duplicates

Two records are one occurrence if they have the same `event_id`, or if they have the same `violation_type` in the same video within 5 seconds, and also the same track or the same plate with status `read`. The earliest record is kept. A plate that is only `uncertain` never merges two events, because it may be wrong.

## Review

`pending` can become `approved` or `rejected`, once, by a named reviewer at a given time.

| Field | Meaning |
| --- | --- |
| `status` | `pending`, `approved` or `rejected` |
| `reviewer` | Who decided |
| `decided_at` | ISO 8601 timestamp |
| `corrected_plate` | The plate as the reviewer read it, if entered |
| `notes` | Free text |

Changing a decision is not part of this contract, because a real review application has to log it as a separate act.

## What is persisted

| Case | Kept | Not kept |
| --- | --- | --- |
| `approved` | The fields of `confirmed_record`, below | Tracks, per-frame reads, and the rule's measurements |
| `rejected` | Event, rule, reviewer, time and notes, for audit | The plate and the images: this vehicle was cleared, so the application should delete its evidence files |
| `pending` | Nothing in the database | |

The plate of a confirmed violation is the reviewer's correction if there is one, else the OCR's text if its status is `read`. A text that is only `uncertain` is never stored unless a reviewer enters it. `plate_source` records which of them it was.

## Database interface

`ViolationStore` is the interface an application implements. `SqliteViolationStore` exists to test it, and is not the application's database.

| Method | Contract |
| --- | --- |
| `save(record)` | Stores a `confirmed_record`. Refuses a record with no reviewer, and an event already stored |
| `get(event_id)` | One confirmed violation, or nothing |
| `find_by_plate(plate_text)` | The confirmed violations of a plate, oldest first |
| `count()` | How many are stored |

Columns: `event_id` (primary key), `video_id`, `violation_type`, `trigger_time_s`, `vehicle_class`, `plate_text`, `plate_source`, `signal_source`, `rule_name`, `rule_version`, `evidence`, `reviewer`, `decided_at`, `notes`.

## Privacy

Records hold real plate numbers, and evidence shows real vehicles. Both stay under `outputs/`, which is git-ignored.

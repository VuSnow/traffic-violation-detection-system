# Traffic Violation Detection System
## Project Overview & Notebook Implementation Specification

> This is the project baseline specification. Notebooks, modules, and reports
> should be checked against it. Changes to scope or decisions should be recorded
> here or in the relevant notebook's conclusion.

### 1. Project Overview

**Project Name:** Traffic Violation Detection System

**Project Type:** Computer Vision / AI Engineering / MLOps

**Primary Language:** Python

**Development Environment:** Jupyter Notebook

**Development Approach:** Prototype First → Validate → Refactor → Integrate → Deploy

### 1.1. Project Purpose

The Traffic Violation Detection System is an end-to-end AI-powered system designed to automatically analyze traffic videos, identify vehicles, track their movements, detect potential traffic violations, recognize license plates, and provide a human verification process before storing confirmed violations.

The system processes video footage from traffic cameras or recorded video files.

**High-level pipeline:**

Video Input → Video Processing → Object Detection → Multi-Object Tracking → Violation Detection → License Plate Recognition → Human Review → Database Storage

The system should be designed with modular components, allowing different detection models, tracking algorithms, and violation detection strategies to be evaluated and replaced independently.

### 1.2. Project Objectives

**Functional Objectives**

1. Read and process traffic videos.
2. Detect vehicles and relevant objects in video frames.
3. Assign tracking IDs to vehicles and track them across multiple frames.
4. Analyze vehicle trajectories and scene conditions to identify potential violations.
5. Detect and recognize license plates associated with suspected violations.
6. Generate reviewable violation events containing visual evidence and metadata.
7. Allow human reviewers to approve or reject detected violations.
8. Persist confirmed violations to a database.

**Engineering & Learning Objectives**

1. Understand each component of a computer vision pipeline rather than treating pretrained models as black boxes.
2. Experiment with pretrained AI models before attempting custom training.
3. Learn how to evaluate model accuracy, limitations, and performance.
4. Practice designing modular AI pipelines with well-defined interfaces.
5. Establish reproducible experiments suitable for future MLOps implementation.
6. Eventually package, deploy, monitor, and improve the complete system.

The goal is both to build a functional system and to understand the engineering decisions behind it.

---

## 2. Development Strategy

The project is divided into two major phases.

### Phase 1 – End-to-End Baseline

Use existing pretrained models and relatively simple, explainable rules to implement a working end-to-end pipeline.

Phase 1 focuses on:

- Understanding the data flow.
- Validating each processing component.
- Identifying integration challenges.
- Creating a minimum viable traffic violation detection system.
- Building a reproducible baseline for future model improvements.

**Training custom deep learning models is not required during Phase 1.**

### Phase 2 – Model Training & Optimization

After the baseline pipeline works:

- Collect and annotate relevant datasets.
- Evaluate baseline model limitations.
- Fine-tune detection models if needed.
- Improve multi-object tracking.
- Improve violation detection rules or introduce learned approaches where justified.
- Improve license plate recognition.
- Compare trained models against the baseline.
- Introduce MLOps practices such as dataset/model versioning, experiment tracking, model deployment, and monitoring.

Phase 2 must build on the results and evidence obtained during Phase 1.

---

## 3. Purpose of Jupyter Notebooks

Jupyter Notebooks are the primary **experimental and educational development environment** during the initial project stages.

They are NOT intended to become the final production application.

### 3.1. Why Notebooks Are Used

Each notebook should focus on one component or closely related group of tasks.

The purpose is to:

1. Understand the theoretical role of the component.
2. Understand its inputs and expected outputs.
3. Implement a minimal working example.
4. Visualize intermediate results.
5. Identify edge cases and technical limitations.
6. Validate whether the component meets its requirements.
7. Record observations, conclusions, and decisions.
8. Prepare validated logic for future extraction into Python modules.

**Development principle:**

Understand → Implement → Visualize → Evaluate → Validate → Refactor

### 3.2. Notebook Requirements

Every notebook must contain:

- **Objective:** What problem is being solved?
- **Background:** Why is this component necessary?
- **Input:** What data does it consume?
- **Output:** What does it produce?
- **Implementation:** Incremental executable Python cells.
- **Visualization:** Relevant visual demonstrations of intermediate results.
- **Validation:** Tests or checks confirming expected behavior.
- **Limitations:** Known failure cases and unresolved questions.
- **Conclusion:** Results and readiness for integration.

Code should be introduced incrementally, not as one large implementation.

Each notebook should be executable from a clean kernel in a fresh environment after dependencies and input data are configured.

The final notebook should not rely on variables that happen to exist because another notebook was previously executed.

When data needs to be shared between notebooks, use clearly documented intermediate files or validated Python modules.

---

## 4. Proposed Notebook Roadmap

The notebook workflow should be implemented in the following order.

### Notebook 00 – Environment Setup & Dataset Exploration

**Objective:** Establish a reproducible working environment and understand the available video dataset.

**Tasks:**

1. Identify and document the Python version and required dependencies.
2. Configure paths for input videos and experimental outputs.
3. Load a sample traffic video.
4. Inspect the video format, duration, resolution, and basic content.
5. Identify what objects and traffic behaviors are visible.
6. Determine whether the video is suitable for experimenting with traffic violation detection.
7. Document environmental assumptions and limitations.

**Expected Outputs:**

- Working Python/Jupyter environment.
- Verified sample video.
- Dataset and environment description.
- Initial observations about the traffic scene.

**Completion Criteria:** The environment can be reproduced and the video can be accessed reliably.

### Notebook 01 – Video Reading & Frame Processing

**Objective:** Understand how digital video is represented and how frames are extracted using OpenCV.

**Topics:**

- Video files and frames.
- FPS (Frames Per Second).
- Frame count.
- Video resolution.
- Frame index and timestamp.
- Sequential and random frame access.

**Tasks:**

1. Open a video using `cv2.VideoCapture`.
2. Validate whether the video was opened successfully.
3. Extract metadata:
   - FPS
   - Total frames
   - Width
   - Height
   - Duration
4. Read a single frame.
5. Inspect frame dimensions and pixel data.
6. Convert BGR to RGB for correct visualization.
7. Visualize frames using Matplotlib.
8. Iterate through multiple frames.
9. Read frames at selected timestamps or frame indices.
10. Handle end-of-video, invalid files, and resource cleanup.

**Expected Outputs:**

- Video metadata.
- Sample frame visualizations.
- Reliable frame iteration logic.
- Understanding of frame index, timestamps, and video position.

**Completion Criteria:** The notebook reliably reads and visualizes video frames without unintentionally skipping frames or mismanaging the capture object's position.

### Notebook 02 – Object Detection

**Objective:** Detect traffic-related objects in video frames using a pretrained object detection model.

**Topics:**

- Object detection fundamentals.
- Bounding boxes.
- Class labels.
- Confidence scores.
- Non-Maximum Suppression (NMS).
- Detection inference.

**Tasks:**

1. Research and select a suitable pretrained detector.
2. Load the pretrained model.
3. Run inference on a single video frame.
4. Inspect raw detection results.
5. Extract bounding boxes, labels, and confidence scores.
6. Filter relevant traffic classes.
7. Draw detection boxes on the frame.
8. Run detection on multiple frames.
9. Evaluate qualitative detection accuracy and obvious failures.
10. Measure approximate inference performance.

**Expected Outputs:**

- Detected objects with coordinates, classes, and confidence scores.
- Annotated video frames.
- Preliminary detection quality and performance observations.

**Completion Criteria:** The notebook consistently detects relevant traffic objects and produces structured detection results that can be consumed by a tracker.

### Notebook 03 – Multi-Object Tracking

**Objective:** Maintain consistent identities for vehicles across successive video frames.

Object detection alone identifies vehicles independently in each frame. Tracking is required to determine how a particular vehicle moves over time.

**Topics:**

- Multi-Object Tracking (MOT).
- Track IDs.
- Data association.
- Bounding box motion.
- Occlusion and lost tracks.
- Track lifecycle.
- Trajectory construction.

**Tasks:**

1. Understand the distinction between detection and tracking.
2. Research an appropriate algorithm (e.g., ByteTrack).
3. Connect object detections to the tracking algorithm.
4. Assign tracking IDs to detected vehicles.
5. Visualize tracking IDs on video frames.
6. Extract vehicle positions across frames.
7. Construct vehicle trajectories.
8. Test tracking through partial occlusion and temporary detection loss.
9. Identify ID switches and fragmentation.
10. Measure tracking performance.

**Expected Outputs:**

- Bounding boxes associated with track IDs.
- Annotated frames showing vehicle identities.
- Per-vehicle trajectories.
- Tracking quality observations.

**Completion Criteria:** The notebook can track vehicles over time and construct usable trajectories, with documented failure cases. A tracking ID is not assumed to remain globally unique or perfectly stable across all conditions.

### Notebook 04 – Traffic Scene Understanding & Configuration

**Objective:** Represent the physical traffic scene using explicit geometric and traffic-rule configurations.

Violation detection often requires knowledge of road geometry rather than object coordinates alone.

**Topics:**

- Regions of Interest (ROI).
- Lane boundaries.
- Stop lines.
- Direction of travel.
- Traffic signal states.
- Camera perspective.

**Tasks:**

1. Inspect the sample traffic scene.
2. Identify relevant lanes, road boundaries, or stop lines.
3. Represent these features using lines and polygons.
4. Define valid vehicle movement directions where appropriate.
5. Visualize scene annotations.
6. Persist scene configurations.
7. Verify that the configurations correspond to the video resolution and camera perspective.

**Expected Outputs:**

- Machine-readable scene configuration.
- Visualized lanes, regions, and relevant boundaries.
- Documented scene assumptions.

**Completion Criteria:** The scene configuration can be loaded and used by subsequent violation detection experiments.

### Notebook 05 – Traffic Violation Detection

**Objective:** Analyze vehicle movement and scene conditions to identify potential traffic violations.

**Potential violation types:**

- Red-light violations.
- Wrong-way driving.
- Lane violations.
- Other violations supported by the available video and reliable scene information.

The initial MVP should implement **one violation type first**. Its selection must depend on the available footage and required observable evidence.

**Tasks:**

1. Select one feasible violation scenario.
2. Define the violation as an explicit, testable rule.
3. Identify all required inputs.
4. Consume vehicle trajectories from the tracking component.
5. Apply scene configurations and relevant traffic conditions.
6. Detect candidate violations.
7. Record when and why a violation was triggered.
8. Visualize detected events.
9. Test valid and invalid scenarios.
10. Identify false positives and false negatives.

For example, red-light detection requires reliable information about the traffic signal state at the time a vehicle crosses the relevant stop line. A trajectory crossing a line alone is insufficient evidence of running a red light.

**Expected Outputs:**

- Candidate violation events.
- Vehicle tracking information associated with each event.
- Timestamps and supporting evidence.
- Rule evaluation results.

**Completion Criteria:** The selected rule reliably produces candidate events in representative scenarios and does not silently treat missing traffic conditions as confirmed violations.

### Notebook 06 – License Plate Detection & Recognition

**Objective:** Extract license plate information from vehicles associated with candidate violations.

**Topics:**

- License plate localization.
- Image cropping.
- Image preprocessing.
- Optical Character Recognition (OCR).
- Recognition confidence.
- Multi-frame recognition.

**Tasks:**

1. Identify the relevant vehicle using its track.
2. Select suitable frames containing that vehicle.
3. Crop or isolate the vehicle region.
4. Detect the license plate within the vehicle region.
5. Crop the license plate.
6. Apply preprocessing when useful.
7. Run OCR to recognize characters.
8. Compare recognition results from multiple frames.
9. Evaluate recognition accuracy and uncertainty.
10. Preserve visual evidence and unresolved recognition results.

**Expected Outputs:**

- License plate image crops.
- Recognized plate text when available.
- Recognition confidence or uncertainty.
- References to the related vehicle track and candidate violation.

**Completion Criteria:** The notebook can process selected vehicle imagery and return structured license plate results, including an explicit unknown/unreadable state.

### Notebook 07 – Violation Event Generation & Review Preparation

**Objective:** Transform model and rule outputs into structured events suitable for human review.

**Tasks:**

1. Define a consistent violation event data structure.
2. Combine violation, tracking, timestamp, and license plate results.
3. Associate evidence images with the event.
4. Prevent duplicate events for the same violation occurrence.
5. Assign an initial review status of `pending`.
6. Save reviewable event records locally.
7. Simulate approval and rejection decisions.
8. Validate which data should be persisted after review.
9. Design the database interface for confirmed violations.

**Expected Outputs:**

- Candidate event records.
- Evidence image references.
- Review status and decision metadata.
- A documented event schema.

**Completion Criteria:** Each candidate event contains sufficient information for later review, and no candidate is automatically treated as a confirmed violation.

The actual review UI and database integration can be developed as separate application components after the experimental contract is validated.

### Notebook 08 – End-to-End Baseline Integration

**Objective:** Validate the complete baseline pipeline from video input through candidate generation and simulated review.

**Tasks:**

1. Load the input video.
2. Extract frames.
3. Run object detection.
4. Update the tracker.
5. Build vehicle trajectories.
6. Apply the selected violation rules.
7. Generate violation candidates.
8. Extract license plate evidence.
9. Create structured review events.
10. Simulate human review and persistence of approved results.
11. Generate an annotated video or visual evidence report.
12. Measure component and overall processing performance.

**Expected Outputs:**

- Annotated video.
- Candidate violation records.
- Extracted evidence.
- Simulated reviewed and confirmed records.
- Performance and execution summary.

**Completion Criteria:** A video can be processed end-to-end using the validated baseline components, with consistent event identities and correct handling of unsuccessful or uncertain stages.

### Notebook 09 – Evaluation & Baseline Analysis

**Objective:** Measure the quality and limitations of the baseline before Phase 2.

**Tasks:**

1. Prepare labeled evaluation samples where feasible.
2. Evaluate object detection quality.
3. Assess tracking stability and ID switches.
4. Measure violation detection precision and recall.
5. Evaluate license plate recognition accuracy on labeled samples.
6. Measure processing time and throughput.
7. Analyze common error patterns.
8. Identify bottlenecks.
9. Document experiments and compare alternatives.
10. Recommend improvements for Phase 2.

**Expected Outputs:**

- Evaluation results.
- Error analysis.
- Baseline performance report.
- Prioritized improvement opportunities.

**Completion Criteria:** The baseline has a documented performance profile, and its limitations are supported by observations or measurements rather than assumptions.

---

## 5. Proposed Repository Organization

```text
traffic-violation-detection-system/
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── 00_environment_and_dataset.ipynb
│   ├── 01_video_reading.ipynb
│   ├── 02_object_detection.ipynb
│   ├── 03_multi_object_tracking.ipynb
│   ├── 04_traffic_scene_configuration.ipynb
│   ├── 05_violation_detection.ipynb
│   ├── 06_license_plate_recognition.ipynb
│   ├── 07_violation_event_generation.ipynb
│   ├── 08_end_to_end_baseline.ipynb
│   └── 09_baseline_evaluation.ipynb
├── configs/
├── data/
│   ├── raw/
│   ├── samples/
│   └── annotations/
├── outputs/
│   ├── frames/
│   ├── videos/
│   ├── evidence/
│   ├── events/
│   └── evaluations/
├── src/
│   └── traffic_violation_detection/
└── tests/
```

This is the proposed structure, not an assertion about the current repository.

Raw video data, trained model weights, generated outputs, and sensitive evidence should not be committed to Git by default.

## 6. Data Contracts Between Components

Before integrating the pipeline, define consistent data structures.

**Frame Data:**
- Video identifier
- Frame index
- Timestamp
- Frame image or reference
- Width and height

**Detection Data:**
- Frame index
- Object class
- Bounding box coordinates
- Detection confidence

**Tracking Data:**
- Video identifier
- Frame index
- Track ID
- Object class
- Bounding box or vehicle position
- Track state
- Timestamp

**Violation Event Data:**
- Event ID
- Video identifier
- Candidate violation type
- Related track ID
- Start/end time or triggering timestamp
- Rule and evidence metadata
- Associated evidence frames
- License plate result, if available
- Review status

**Review Data:**
- Event ID
- Review decision
- Reviewer identifier
- Review timestamp
- Optional corrected plate text or notes

Track IDs must be scoped to a particular video or tracking session. Event IDs must remain distinct from track IDs.

Use consistent coordinate formats, timestamp conventions, and JSON-serializable metadata across notebook boundaries.

## 7. Instructions for the AI Coding Agent

### 7.1. General Implementation Rules

1. Treat this specification as the project baseline.
2. Inspect the existing repository before creating or modifying files.
3. Do not assume a dependency or dataset already exists.
4. Preserve existing useful code unless a rewrite is necessary and justified.
5. Start from Notebook 00 and progress sequentially.
6. Keep notebook cells small, readable, and executable.
7. Include Markdown explanations before major implementation steps.
8. Visualize computer vision outputs whenever doing so assists validation.
9. Do not build complex abstractions before validating the underlying logic.
10. Do not invent experimental results, evaluation metrics, or successful executions.

### 7.2. Implementation Workflow

For every notebook:

1. Explain the component's role in the system.
2. Define the notebook's input/output contract.
3. Introduce the relevant concepts.
4. Implement the smallest working example.
5. Verify the output.
6. Extend the implementation incrementally.
7. Explore edge cases.
8. Record measured results and limitations.
9. Identify dependencies for the next notebook.
10. Summarize whether the component is ready for integration.

### 7.3. Refactoring Strategy

Do not immediately convert every experiment into production-level Python classes.

During notebook development:

- Prefer understandable functions and simple data structures.
- Optimize for clarity and verification.
- Avoid premature complexity.

After a component is validated:

- Move reusable logic to the `src/` directory.
- Add automated tests.
- Keep notebooks as experimentation and demonstration clients.
- Avoid duplicating business logic across notebooks and Python modules.
- Preserve the ability to rerun experiments.

### 7.4. Agent Reporting Requirements

After completing each notebook, report:

- What was implemented.
- Which concepts were demonstrated.
- What data or files were used.
- Which cells/tests were actually executed.
- What results were observed.
- Which issues remain unresolved.
- Whether the acceptance criteria were satisfied.
- What should be developed next.

Clearly distinguish verified results from assumptions or proposed improvements.

---

## 8. Definition of Done for Phase 1

Phase 1 is complete when:

- The system can read and process a traffic video.
- Relevant traffic objects can be detected.
- Vehicles can be tracked across frames.
- At least one clearly defined violation type can be evaluated.
- Potential violations can be represented as candidate events.
- License plate recognition is attempted for relevant vehicle evidence.
- Candidate events can be reviewed and approved or rejected.
- Confirmed events can be persisted through a validated database integration.
- An end-to-end execution has been demonstrated.
- Important failure cases and baseline performance have been documented.

A successful notebook experiment alone is not sufficient to claim that the complete production system is finished.

## 9. Final Project Principle

**The objective is not simply to connect pretrained models and obtain a result.**

The project should develop an explainable, testable, reproducible, and extensible understanding of how an AI-powered traffic violation detection system works.

Each notebook must contribute both:

1. **Engineering knowledge:** Understanding the component, its decisions, and its limitations.
2. **Technical progress:** Producing validated functionality that can be integrated into the final system.

The eventual system should be developed from validated components, with human review remaining the final verification step for reported violations.

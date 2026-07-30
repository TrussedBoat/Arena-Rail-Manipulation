# YOLO + VLM Hybrid Orchestrator Implementation Pipeline

## Purpose and Scope

This document defines the planned implementation for replacing the existing navigation and pickup-object localization phase with a hybrid architecture:

- A Vision-Language Model (VLM) remains the high-level LangGraph orchestrator.
- A deterministic Python tool performs active search, YOLO detection, visual centering, and 1D localization.
- Existing pick, place, homing, ROS 2 topics, Isaac Sim configuration, and Bash workflows remain unchanged.
- Future implementation changes are restricted to `main.py`, `agent.py`, and `tools.py`.

No automatic model-download behavior will be introduced.

## Recommended VLM

Use **Qwen2.5-VL-7B-Instruct Q4_K_M GGUF** with the **Q8_0 multimodal projector**:

- `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
- `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf`

This is the preferred first implementation because it offers a strong balance of:

- Mature multimodal support in `llama.cpp`.
- OpenAI-compatible serving and tool calling.
- Stronger instruction following and tool sequencing than the 3B variant.
- Considerably lower latency and VRAM use than the current 35B configuration.
- Compatibility with the target GPU allocation: up to 8GB of the GPU's 16GB total VRAM is reserved for the VLM, leaving the other 8GB for Isaac Sim.

The approximate static model allocation is 4.68GB for the Q4_K_M language model plus 853MB for the Q8_0 projector, or about 5.53GB combined. A 4096-token KV cache is expected to add approximately 225MB, leaving roughly 2.2GB of the dedicated 8GB VLM budget for CUDA and computation buffers. Actual peak VLM VRAM must remain at or below 8GB, and the `< 5 seconds` latency requirement must be verified on the target GPU while Isaac Sim is running in its separate 8GB allocation.

**Qwen2-VL-7B-Instruct Q4_K_M** is the fallback if the selected llama.cpp build has a compatibility regression with Qwen2.5-VL. Qwen2.5-VL-7B remains the primary recommendation because of its stronger instruction following and newer vision-language behavior.

The GGUF path, mmproj path, llama.cpp executable path, model alias, and server options will be supplied through environment-based configuration. The startup process will fail clearly if a configured file does not exist; it will never download missing artifacts.

## Recommended YOLO Baseline

Use a locally supplied **`yolo11s.pt`** checkpoint as the initial detector.

YOLO11s is small enough for frequent stop-and-go inference while providing a better accuracy margin than the nano variant. Its COCO-pretrained vocabulary includes `apple`. The checkpoint path will be configurable, and the tool will verify that the normalized target label exists in the checkpoint's class vocabulary before moving the robot.

The live Isaac Sim test remains the final authority for whether the pretrained detector reliably exceeds the required confidence threshold.

## Implementation Pipeline

### 1. Establish Runtime Configuration and Validation

Add a centralized configuration-loading layer within the permitted Python files. It will cover:

- llama.cpp executable, GGUF, and mmproj paths.
- Server host, port, model alias, and 4096-token context size.
- YOLO checkpoint path and inference settings.
- Physical rail minimum and maximum positions.
- Rail waypoint spacing and centering step limits.
- Wrist-search angles and the calibrated final centering angle.
- Motion-settling duration and joint-position tolerances.
- Horizontal centering tolerance.
- Dynamic semantic-coordinate JSON path.

All paths and numeric limits will be validated before motion begins. Missing model files, invalid rail bounds, unavailable target classes, or an unresponsive server will produce explicit failures. No model auto-download scripts or implicit downloads will be allowed.

### 2. Reconfigure the llama.cpp Server Lifecycle

Preserve the existing `tmux`-managed, OpenAI-compatible `llama-server` architecture while replacing hard-coded model paths and oversized settings.

The planned server configuration will use:

- Qwen2.5-VL-7B-Instruct Q4_K_M.
- The matching Q8_0 mmproj file.
- A 4096-token context.
- One inference slot.
- GPU offload and flash attention when supported.
- Tool-compatible Jinja chat-template handling.
- A short completion allowance, such as 128–256 tokens, because orchestration responses should consist primarily of tool calls.
- Native projector preprocessing for any VLM image input.

Startup validation will include port readiness, a short warm-up request, and timing capture. The `< 5 seconds` requirement will be treated as a live acceptance gate rather than inferred from VRAM capacity alone.

### 3. Minimize the LangGraph Tool Surface and Prompt

Keep LangGraph and the VLM as the high-level planner, but expose only these tools:

1. `start_joint_controller`
2. `search_and_locate_with_yolo`
3. `execute_pick_script`
4. `move_rail_to_object`
5. `execute_place_script`
6. `finish_task`

Image capture, raw joint-state inspection, wrist turning, rail stepping, YOLO inference, and JSON persistence will become internal implementation details rather than VLM-visible tools.

The system prompt will be shortened to enforce this sequence:

```text
initialize
  -> normalize pickup label
  -> search and localize pickup with YOLO
  -> execute existing pick workflow
  -> move to the fixed bowl
  -> execute existing place workflow
  -> return home
  -> finish task
```

The VLM will interpret the user's natural-language command and pass a canonical target such as `apple` to the search tool. It will not select waypoints, evaluate detections, calculate servo movements, or inspect camera frames during active search.

`move_rail_to_object` will remain available for static destinations such as the fixed bowl and the home position. It will no longer provide pickup-object localization.

### 4. Build the Deterministic YOLO Detection Contract

Create an internal detector component used exclusively by the search tool. For each stationary camera observation it will:

1. Obtain the latest wrist-camera frame from the existing ROS 2 interface.
2. Decode the image without sending it to the VLM.
3. Run the configured pretrained YOLO model.
4. Normalize detector labels before comparison.
5. Retain only detections matching the requested canonical label.
6. Reject bounding boxes that are not completely inside the image border.
7. Return the label, confidence, pixel bounding box, area, center point, and normalized horizontal-center error.

If multiple target instances are present, the instance with the highest confidence will be selected. Bounding-box area will be used as the tie-breaker.

A candidate below or equal to `0.85` may guide centering, but it cannot be accepted as a successful localization. Final success requires confidence strictly greater than `0.85`.

### 5. Implement the Stop-and-Go Active Search State Machine

Encapsulate the entire search process inside `search_and_locate_with_yolo(target_object)`. The VLM will make one blocking tool call and receive one structured success or failure result.

The planned state flow is:

```text
INITIALIZE
    |
    v
MOVE_TO_RAIL_WAYPOINT
    |
    v
WAIT_UNTIL_STOPPED
    |
    v
MOVE_WRIST_TO_SEARCH_ANGLE
    |
    v
WAIT_UNTIL_STOPPED
    |
    v
CAPTURE_AND_RUN_YOLO
    |                         |
    | no target               | target candidate
    v                         v
NEXT ANGLE / WAYPOINT     HALT_AND_CENTER
    |                         |
    | rail exhausted          | centered, valid, > 0.85
    v                         v
FAILURE                   SAVE_AND_SUCCEED
```

Starting from the current rail position, the state machine will scan toward one physical rail limit and then across to the opposite limit, ensuring that the full rail length is covered. At each rail waypoint, `panda_joint1` will sweep through configured left, center, and right camera angles.

The state machine will never evaluate YOLO frames while `rail_j1` or `panda_joint1` is moving. It will wait for joint convergence and a settling interval before acquiring a fresh frame.

The existing ROS position-command interface does not directly control velocity. Moderate search motion will therefore be achieved through conservative waypoint spacing, bounded increments, complete stop verification, and the existing controller's configured velocity behavior. ROS topics and controller configuration will not be changed.

### 6. Add Rail-Based Visual Centering and Lost-Target Recovery

When YOLO reports a matching candidate, rail search will stop immediately and transition to the approach/centering state.

Centering will operate as follows:

1. Move `panda_joint1` to one calibrated final camera angle.
2. Wait until the wrist is stationary.
3. Calculate horizontal error between the bounding-box center and image center.
4. Convert the normalized error into a bounded `rail_j1` correction.
5. Move the rail and wait until it has stopped.
6. Run YOLO on a new stationary frame.
7. Repeat until the object is centered and the detection satisfies the final contract.

Using one fixed final wrist angle is essential because the saved coordinate is derived solely from the rail position. Allowing different wrist angles at final localization would make the 1D coordinate dependent on camera yaw.

If the target disappears during centering, the tool will backtrack slightly toward the last rail position where the target was observed, stop, and attempt reacquisition. It will not use continuous tracking, optical flow, depth, ROS TF, or 6-DoF pose estimation.

The success condition will be all of the following on the same stationary frame:

- Canonical label matches the requested target.
- Confidence is strictly greater than `0.85`.
- Bounding box is completely inside the image.
- Horizontal center error is within the configured tolerance.

If reacquisition fails, the state machine will resume or terminate according to the remaining unscanned rail range. Scanning the complete rail without a valid centered detection returns failure.

### 7. Calculate and Persist the Session-Relative Coordinate

After successful centering, calculate the target's 1D coordinate as:

```text
object_x = current_absolute_rail_j1 - initial_rail_position
```

The first valid `rail_j1` telemetry value in the session remains the reference origin. No ROS TF, depth, camera calibration matrix, or Cartesian pose estimate will be introduced.

Write the result to a new file named `semantic_distances_dynamic.json`, keyed by canonical label. A subsequent localization of the same label will overwrite that label's existing entry without deleting other labels.

The persistent spatial record will contain the 1D `x` coordinate. Detection confidence and bounding-box details will be included in the tool's structured return value and logs, but will not be treated as additional spatial dimensions.

Example logical shape:

```json
{
  "apple": {
    "x": 1.234
  }
}
```

The saved coordinate remains valid for the current session origin and supports the subsequent classical pickup workflow.

### 8. Preserve and Integrate Existing Manipulation Workflows

Do not modify:

- `rail_demo_pick.sh`
- `rail_demo_place.sh`
- ROS 2 topic names or message types
- Isaac Sim world or robot configuration
- Existing controller implementations
- Classical pick, place, or homing behavior

The search tool will finish with the rail centered at the dynamically located pickup object. The orchestrator will then invoke the existing pick script with the canonical label. After successful pickup, it will use the existing static navigation mechanism for the fixed bowl, invoke the existing place script, return the rail and arm to the established home state, and call `finish_task`.

Tool results will use structured statuses such as `success`, `failure`, `reason`, `target`, `confidence`, `bbox`, and `x` so LangGraph can stop safely instead of attempting pick or place after a failed search.

### 9. Validate in the Live Isaac Sim Arena

Validation will be performed exclusively in the live simulation arena. The primary acceptance scenario is:

1. Randomly place an apple along the table.
2. Start the orchestrator with a natural-language pickup-and-place command.
3. Confirm that the VLM normalizes the target to `apple` and calls the deterministic search tool once.
4. Observe a stop-and-go rail sweep and wrist-camera search.
5. Confirm that no YOLO frames are processed during joint motion.
6. Confirm immediate transition from detection to centering.
7. Test lost-target backtracking during centering.
8. Verify confidence, border, and centering acceptance checks.
9. Compare the saved `x` coordinate with the final rail displacement from the session origin.
10. Complete the unchanged pick, fixed-bowl navigation, place, and home workflows.

Additional failure tests will cover:

- Target absent across the full rail length.
- Only low-confidence detections available.
- Bounding box touching an image border.
- Multiple apple instances.
- Temporary target loss during centering.
- Missing or invalid model paths.
- YOLO checkpoint without the requested class.
- Joint motion timeout.
- VLM inference exceeding five seconds.
- Peak combined VRAM pressure while Isaac Sim, llama.cpp, and YOLO are active.

Final acceptance requires repeatable success across randomized apple positions, VLM inference below five seconds on the target machine, and no regressions in the existing manipulation and homing workflows.

## Planned File Ownership

| File | Planned responsibility |
|---|---|
| `main.py` | Runtime startup, concise system prompt, task input, session lifecycle, and cleanup |
| `agent.py` | Minimal tool schemas, LangGraph routing, structured tool results, and completion control |
| `tools.py` | Config validation, llama.cpp lifecycle, YOLO detector, active-search state machine, centering, persistence, and existing script wrappers |

No implementation changes are planned outside these three Python files.

## Review Gates Before Coding

Implementation should begin only after approval of:

1. Qwen2.5-VL-7B-Instruct Q4_K_M plus Q8_0 mmproj as the initial VLM, with Qwen2-VL-7B-Instruct Q4_K_M retained only as a compatibility fallback.
2. `yolo11s.pt` as the initial pretrained detector.
3. The candidate rule: any matching candidate stops search, but only a centered detection with confidence `> 0.85` may be saved.
4. A single calibrated `panda_joint1` angle for final rail-based centering.
5. Environment-based configuration without automatic model downloads.
6. The nine-stage pipeline and live-arena acceptance criteria above.

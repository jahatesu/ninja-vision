import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.hand_tracker import HandTracker


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HAND_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

POSE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pose_landmarker.task"
)


# =========================================================
# Hand connections
# =========================================================

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


# =========================================================
# Pose landmark indices
# =========================================================

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16


# =========================================================
# Drawing functions
# =========================================================

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw hand bones.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    # Draw hand joints.
    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )


def landmark_to_pixel(
    landmark,
    width,
    height,
):
    return (
        int(landmark.x * width),
        int(landmark.y * height),
    )


def draw_arm(
    frame,
    pose_landmarks,
    shoulder_index,
    elbow_index,
    wrist_index,
):
    height, width, _ = frame.shape

    shoulder = landmark_to_pixel(
        pose_landmarks[shoulder_index],
        width,
        height,
    )

    elbow = landmark_to_pixel(
        pose_landmarks[elbow_index],
        width,
        height,
    )

    wrist = landmark_to_pixel(
        pose_landmarks[wrist_index],
        width,
        height,
    )

    cv2.line(
        frame,
        shoulder,
        elbow,
        (255, 255, 0),
        2,
    )

    cv2.line(
        frame,
        elbow,
        wrist,
        (255, 255, 0),
        2,
    )

    cv2.circle(
        frame,
        shoulder,
        5,
        (255, 255, 0),
        -1,
    )

    cv2.circle(
        frame,
        elbow,
        5,
        (255, 255, 0),
        -1,
    )

    # Make pose wrist particularly visible.
    cv2.circle(
        frame,
        wrist,
        8,
        (255, 0, 255),
        -1,
    )


def status_color(status):
    if status == "visible":
        return (0, 255, 0)

    if status == "occluded":
        return (0, 255, 255)

    return (0, 0, 255)


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Check models
    # -----------------------------------------------------

    if not HAND_MODEL_PATH.exists():
        print("ERROR: Hand model not found:")
        print(HAND_MODEL_PATH)
        return

    if not POSE_MODEL_PATH.exists():
        print("ERROR: Pose model not found:")
        print(POSE_MODEL_PATH)
        return

    # -----------------------------------------------------
    # Hand Landmarker
    # -----------------------------------------------------

    hand_base_options = python.BaseOptions(
        model_asset_path=str(
            HAND_MODEL_PATH
        )
    )

    hand_options = vision.HandLandmarkerOptions(
        base_options=hand_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_landmarker = (
        vision.HandLandmarker.create_from_options(
            hand_options
        )
    )

    # -----------------------------------------------------
    # Pose Landmarker
    # -----------------------------------------------------

    pose_base_options = python.BaseOptions(
        model_asset_path=str(
            POSE_MODEL_PATH
        )
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=pose_base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    pose_landmarker = (
        vision.PoseLandmarker.create_from_options(
            pose_options
        )
    )

    # -----------------------------------------------------
    # Temporal tracker
    # -----------------------------------------------------

    hand_tracker = HandTracker(
        grace_period=0.35
    )

    # -----------------------------------------------------
    # Camera
    # -----------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not access webcam.")

        hand_landmarker.close()
        pose_landmarker.close()

        return

    print()
    print("==============================")
    print("       NINJA VISION")
    print("      HYBRID TRACKING")
    print("==============================")
    print()
    print("Hand tracking + pose tracking")
    print("are now running together.")
    print()
    print("Try overlapping your hands.")
    print("Press Q to quit.")
    print()

    start_time = time.perf_counter()

    # FPS calculation
    fps = 0.0
    previous_frame_time = time.perf_counter()

    # -----------------------------------------------------
    # Camera loop
    # -----------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        frame = cv2.flip(
            frame,
            1
        )

        height, width, _ = frame.shape

        # -------------------------------------------------
        # Convert image
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )

        # -------------------------------------------------
        # Run BOTH models
        # -------------------------------------------------

        hand_result = (
            hand_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        pose_result = (
            pose_landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        # -------------------------------------------------
        # Organize hand detections
        # -------------------------------------------------

        detected_hands = {
            "left": None,
            "right": None,
        }

        for hand_index, hand_landmarks in enumerate(
            hand_result.hand_landmarks
        ):

            if hand_index >= len(
                hand_result.handedness
            ):
                continue

            classification = (
                hand_result
                .handedness[hand_index][0]
            )

            hand_name = (
                classification
                .category_name
                .lower()
            )

            if hand_name == "left":
                detected_hands["left"] = (
                    hand_landmarks
                )

            elif hand_name == "right":
                detected_hands["right"] = (
                    hand_landmarks
                )

        # -------------------------------------------------
        # Temporal tracking
        # -------------------------------------------------

        tracked_hands = hand_tracker.update(
            detected_hands
        )

        left_status = (
            tracked_hands["left"]["status"]
        )

        right_status = (
            tracked_hands["right"]["status"]
        )

        # -------------------------------------------------
        # Draw visible hand landmarks
        # -------------------------------------------------

        for hand_name in (
            "left",
            "right",
        ):

            hand_data = tracked_hands[
                hand_name
            ]

            if (
                hand_data["status"]
                == "visible"
            ):
                draw_hand(
                    frame,
                    hand_data["landmarks"],
                )

        # -------------------------------------------------
        # Pose information
        # -------------------------------------------------

        pose_detected = False

        left_pose_wrist_visible = False
        right_pose_wrist_visible = False

        if pose_result.pose_landmarks:

            pose_detected = True

            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )

            # Draw arms.
            draw_arm(
                frame,
                pose_landmarks,
                LEFT_SHOULDER,
                LEFT_ELBOW,
                LEFT_WRIST,
            )

            draw_arm(
                frame,
                pose_landmarks,
                RIGHT_SHOULDER,
                RIGHT_ELBOW,
                RIGHT_WRIST,
            )

            left_wrist = pose_landmarks[
                LEFT_WRIST
            ]

            right_wrist = pose_landmarks[
                RIGHT_WRIST
            ]

            # Pose landmarks expose visibility.
            left_pose_wrist_visible = (
                left_wrist.visibility > 0.5
            )

            right_pose_wrist_visible = (
                right_wrist.visibility > 0.5
            )

        # -------------------------------------------------
        # Overall interpretation
        # -------------------------------------------------

        if (
            left_status == "visible"
            and right_status == "visible"
        ):
            system_state = (
                "FULL HAND TRACKING"
            )

        elif (
            pose_detected
            and (
                left_pose_wrist_visible
                or right_pose_wrist_visible
            )
        ):
            system_state = (
                "PARTIAL HAND TRACKING "
                "+ POSE SUPPORT"
            )

        else:
            system_state = (
                "LIMITED TRACKING"
            )

        # -------------------------------------------------
        # FPS
        # -------------------------------------------------

        current_frame_time = (
            time.perf_counter()
        )

        delta = (
            current_frame_time
            - previous_frame_time
        )

        if delta > 0:
            current_fps = 1.0 / delta

            # Smooth the displayed FPS.
            fps = (
                0.9 * fps
                + 0.1 * current_fps
            )

        previous_frame_time = (
            current_frame_time
        )

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        cv2.putText(
            frame,
            "NINJA VISION // HYBRID TRACKING",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"LEFT HAND: {left_status.upper()}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color(left_status),
            2,
        )

        cv2.putText(
            frame,
            f"RIGHT HAND: {right_status.upper()}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color(right_status),
            2,
        )

        pose_text = (
            "VISIBLE"
            if pose_detected
            else "MISSING"
        )

        cv2.putText(
            frame,
            f"POSE: {pose_text}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                (0, 255, 0)
                if pose_detected
                else (0, 0, 255)
            ),
            2,
        )

        cv2.putText(
            frame,
            system_state,
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "GREEN=HAND  CYAN=ARMS  MAGENTA=POSE WRISTS",
            (20, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "[Q] QUIT",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        cv2.imshow(
            "Ninja Vision - Hybrid Tracking",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    camera.release()

    hand_landmarker.close()
    pose_landmarker.close()

    cv2.destroyAllWindows()

    print()
    print(
        "Ninja Vision hybrid tracking stopped."
    )


if __name__ == "__main__":
    main()
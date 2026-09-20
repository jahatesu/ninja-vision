import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.hand_tracker import HandTracker
from utils.landmark_utils import create_gesture_features


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

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


# =========================================================
# Configuration
# =========================================================

SUPPORTED_GESTURES = {
    "bird",
    "boar",
    "dog",
    "dragon",
    "hare",
    "horse",
    "monkey",
    "ox",
    "ram",
    "rat",
    "snake",
    "tiger",
    "shadow_clone",
    "none",
}

FEATURE_COUNT = 246

AUTO_CAPTURE_INTERVAL = 0.20

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# =========================================================
# Arguments
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ninja Vision dataset collector."
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(SUPPORTED_GESTURES),
        help="Gesture class to collect.",
    )

    return parser.parse_args()


# =========================================================
# CSV utilities
# =========================================================

def get_existing_sample_count(file_path):
    """
    Return the number of existing data rows in a CSV.
    The header is not counted.
    """

    if not file_path.exists():
        return 0

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)

        rows = list(reader)

    if not rows:
        return 0

    return max(0, len(rows) - 1)


def save_sample(
    file_path,
    label,
    features,
):
    """
    Append one training sample.
    """

    if len(features) != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features, "
            f"received {len(features)}."
        )

    file_exists = file_path.exists()

    with file_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"]

            header.extend(
                f"feature_{index}"
                for index in range(FEATURE_COUNT)
            )

            writer.writerow(header)

        writer.writerow(
            [label] + features
        )


def undo_last_sample(file_path):
    """
    Remove the final data row from the CSV.

    Returns True if a sample was removed.
    """

    if not file_path.exists():
        return False

    with file_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.reader(file))

    # Header only, or empty file.
    if len(rows) <= 1:
        return False

    rows.pop()

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    return True


def reset_dataset(file_path):
    """
    Delete the current gesture CSV.
    """

    if file_path.exists():
        file_path.unlink()


# =========================================================
# Drawing
# =========================================================

def draw_hand(
    frame,
    landmarks,
):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        points.append(
            (
                int(landmark.x * width),
                int(landmark.y * height),
            )
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1,
        )


def status_color(status):
    if status == "visible":
        return (0, 255, 0)

    if status == "occluded":
        return (0, 255, 255)

    return (0, 0, 255)


# =========================================================
# Quality assessment
# =========================================================

def assess_sample_quality(
    label,
    left_status,
    right_status,
    pose_landmarks,
):
    """
    Determine whether the current frame is suitable
    for dataset collection.

    Returns:
        quality
        message
        can_capture
    """

    pose_available = (
        pose_landmarks is not None
    )

    visible_hands = sum(
        status == "visible"
        for status in (
            left_status,
            right_status,
        )
    )

    tracked_hands = sum(
        status in {
            "visible",
            "occluded",
        }
        for status in (
            left_status,
            right_status,
        )
    )

    # -----------------------------------------------------
    # Negative class
    # -----------------------------------------------------

    if label == "none":

        if visible_hands >= 1 or pose_available:
            return (
                "READY",
                "Negative example",
                True,
            )

        return (
            "WAIT",
            "Enter camera view",
            False,
        )

    # -----------------------------------------------------
    # Actual Naruto seals
    # -----------------------------------------------------

    if visible_hands == 2:
        return (
            "EXCELLENT",
            "Both hands visible",
            True,
        )

    if (
        visible_hands == 1
        and tracked_hands == 2
        and pose_available
    ):
        return (
            "OCCLUDED",
            "Temporary hand occlusion",
            True,
        )

    if (
        visible_hands == 1
        and pose_available
    ):
        return (
            "PARTIAL",
            "One hand + pose support",
            False,
        )

    return (
        "WAIT",
        "Need better hand visibility",
        False,
    )


# =========================================================
# Main
# =========================================================

def main():

    args = parse_arguments()
    label = args.label

    # -----------------------------------------------------
    # Verify models
    # -----------------------------------------------------

    if not HAND_MODEL_PATH.exists():
        print("ERROR: Hand model missing:")
        print(HAND_MODEL_PATH)
        return

    if not POSE_MODEL_PATH.exists():
        print("ERROR: Pose model missing:")
        print(POSE_MODEL_PATH)
        return

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        DATA_DIR
        / f"{label}.csv"
    )

    sample_count = get_existing_sample_count(
        output_file
    )

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

    # -----------------------------------------------------
    # Collector state
    # -----------------------------------------------------

    auto_capture = False

    last_auto_capture = 0.0

    last_capture_message = ""
    last_capture_message_time = 0.0

    start_time = time.perf_counter()

    print()
    print("================================")
    print("         NINJA VISION")
    print("       DATA COLLECTOR")
    print("================================")
    print()
    print(f"Gesture: {label.upper()}")
    print(f"Existing samples: {sample_count}")
    print()
    print("SPACE = Capture")
    print("A     = Toggle auto capture")
    print("U     = Undo last sample")
    print("R     = Reset current dataset")
    print("Q     = Quit")
    print()

    # -----------------------------------------------------
    # Camera loop
    # -----------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera.")
            break

        frame = cv2.flip(
            frame,
            1,
        )

        height, width, _ = frame.shape

        # -------------------------------------------------
        # Convert frame
        # -------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
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
        # Run models
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
        # Current hand detections
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
        # IMPORTANT:
        # distinguish CURRENT landmarks from cached ones.
        # -------------------------------------------------

        left_current = detected_hands["left"]
        right_current = detected_hands["right"]

        # -------------------------------------------------
        # Pose
        # -------------------------------------------------

        pose_landmarks = None

        if pose_result.pose_landmarks:
            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )

        # -------------------------------------------------
        # Draw current hands only
        # -------------------------------------------------

        if left_current is not None:
            draw_hand(
                frame,
                left_current,
            )

        if right_current is not None:
            draw_hand(
                frame,
                right_current,
            )

        # -------------------------------------------------
        # Quality
        # -------------------------------------------------

        (
            quality,
            quality_message,
            can_capture,
        ) = assess_sample_quality(
            label=label,
            left_status=left_status,
            right_status=right_status,
            pose_landmarks=pose_landmarks,
        )

        # -------------------------------------------------
        # Function for current sample
        # -------------------------------------------------

        def capture_current_sample():
            nonlocal sample_count
            nonlocal last_capture_message
            nonlocal last_capture_message_time

            if not can_capture:
                last_capture_message = (
                    "NOT CAPTURED - LOW QUALITY"
                )

                last_capture_message_time = (
                    time.perf_counter()
                )

                return

            # ---------------------------------------------
            # For visible hands use current landmarks.
            #
            # For a temporarily occluded hand, use cached
            # landmarks but preserve its OCCLUDED status.
            # ---------------------------------------------

            left_features_hand = left_current
            right_features_hand = right_current

            if (
                left_features_hand is None
                and left_status == "occluded"
            ):
                left_features_hand = (
                    tracked_hands[
                        "left"
                    ]["landmarks"]
                )

            if (
                right_features_hand is None
                and right_status == "occluded"
            ):
                right_features_hand = (
                    tracked_hands[
                        "right"
                    ]["landmarks"]
                )

            features = create_gesture_features(
                left_hand=left_features_hand,
                right_hand=right_features_hand,
                left_status=left_status,
                right_status=right_status,
                pose_landmarks=pose_landmarks,
            )

            save_sample(
                output_file,
                label,
                features,
            )

            sample_count += 1

            last_capture_message = (
                f"CAPTURED #{sample_count}"
            )

            last_capture_message_time = (
                time.perf_counter()
            )

            print(
                f"Captured {label}: "
                f"{sample_count}"
            )

        # -------------------------------------------------
        # Auto capture
        # -------------------------------------------------

        current_time = time.perf_counter()

        if (
            auto_capture
            and can_capture
            and (
                current_time
                - last_auto_capture
                >= AUTO_CAPTURE_INTERVAL
            )
        ):
            capture_current_sample()

            last_auto_capture = (
                current_time
            )

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        cv2.putText(
            frame,
            "NINJA VISION // DATA COLLECTOR",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"GESTURE: {label.upper()}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"SAMPLES: {sample_count}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # Hand statuses
        cv2.putText(
            frame,
            f"LEFT: {left_status.upper()}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color(left_status),
            2,
        )

        cv2.putText(
            frame,
            f"RIGHT: {right_status.upper()}",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color(right_status),
            2,
        )

        pose_status = (
            "VISIBLE"
            if pose_landmarks is not None
            else "MISSING"
        )

        cv2.putText(
            frame,
            f"POSE: {pose_status}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                (0, 255, 0)
                if pose_landmarks is not None
                else (0, 0, 255)
            ),
            2,
        )

        # Quality
        quality_color = (
            (0, 255, 0)
            if can_capture
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            f"QUALITY: {quality}",
            (20, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            quality_color,
            2,
        )

        cv2.putText(
            frame,
            quality_message,
            (20, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # Auto capture
        auto_text = (
            "ON"
            if auto_capture
            else "OFF"
        )

        cv2.putText(
            frame,
            f"AUTO CAPTURE: {auto_text}",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                (0, 255, 0)
                if auto_capture
                else (255, 255, 255)
            ),
            2,
        )

        # Temporary capture message
        if (
            current_time
            - last_capture_message_time
            < 1.0
        ):
            cv2.putText(
                frame,
                last_capture_message,
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # Controls
        cv2.putText(
            frame,
            "[SPACE] CAPTURE  [A] AUTO",
            (20, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "[U] UNDO  [R] RESET  [Q] QUIT",
            (20, height - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        cv2.imshow(
            "Ninja Vision - Dataset Collector",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        # -------------------------------------------------
        # Keyboard controls
        # -------------------------------------------------

        if key == ord(" "):
            capture_current_sample()

        elif key == ord("a"):
            auto_capture = not auto_capture

            # Prevent an immediate stale capture.
            last_auto_capture = (
                time.perf_counter()
            )

            print(
                "Auto capture:",
                "ON"
                if auto_capture
                else "OFF",
            )

        elif key == ord("u"):

            if undo_last_sample(
                output_file
            ):
                sample_count = max(
                    0,
                    sample_count - 1,
                )

                print(
                    "Removed last sample."
                )

                last_capture_message = (
                    "LAST SAMPLE REMOVED"
                )

            else:
                print(
                    "No sample available to undo."
                )

                last_capture_message = (
                    "NOTHING TO UNDO"
                )

            last_capture_message_time = (
                time.perf_counter()
            )

        elif key == ord("r"):

            # Safety:
            # first press disables auto capture.
            if auto_capture:
                auto_capture = False

                last_capture_message = (
                    "AUTO OFF - PRESS R AGAIN TO RESET"
                )

                last_capture_message_time = (
                    time.perf_counter()
                )

                print(
                    "Auto capture disabled. "
                    "Press R again to reset."
                )

            else:
                reset_dataset(
                    output_file
                )

                sample_count = 0

                last_capture_message = (
                    "DATASET RESET"
                )

                last_capture_message_time = (
                    time.perf_counter()
                )

                print(
                    f"{label} dataset reset."
                )

        elif key == ord("q"):
            break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    camera.release()

    hand_landmarker.close()
    pose_landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("==============================")
    print("Collection finished.")
    print(f"Gesture: {label}")
    print(f"Total samples: {sample_count}")
    print(f"Saved to: {output_file}")
    print("==============================")


if __name__ == "__main__":
    main()
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


# =========================================================
# Configuration
# =========================================================

EXPECTED_FEATURE_COUNT = 246


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Check models
    # -----------------------------------------------------

    if not HAND_MODEL_PATH.exists():
        print("ERROR: Hand model missing.")
        print(HAND_MODEL_PATH)
        return

    if not POSE_MODEL_PATH.exists():
        print("ERROR: Pose model missing.")
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
    # Temporal hand tracker
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
    print("================================")
    print("         NINJA VISION")
    print("      FEATURE VECTOR TEST")
    print("================================")
    print()
    print("Expected features:", EXPECTED_FEATURE_COUNT)
    print()
    print("Show your hands to the camera.")
    print("Press SPACE to inspect the current vector.")
    print("Press Q to quit.")
    print()

    start_time = time.perf_counter()

    last_message = "WAITING"

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
        # Run both models
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
        # Current hands
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
                hand_result.handedness[
                    hand_index
                ][0]
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
        # Tracking state
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

        left_hand = detected_hands["left"]
        right_hand = detected_hands["right"]

        # Use short-term cached landmarks for occlusion.
        if (
            left_hand is None
            and left_status == "occluded"
        ):
            left_hand = (
                tracked_hands["left"]["landmarks"]
            )

        if (
            right_hand is None
            and right_status == "occluded"
        ):
            right_hand = (
                tracked_hands["right"]["landmarks"]
            )

        # -------------------------------------------------
        # Pose
        # -------------------------------------------------

        pose_landmarks = None

        if pose_result.pose_landmarks:
            pose_landmarks = (
                pose_result.pose_landmarks[0]
            )

        # -------------------------------------------------
        # Create feature vector
        # -------------------------------------------------

        try:
            features = create_gesture_features(
                left_hand=left_hand,
                right_hand=right_hand,
                left_status=left_status,
                right_status=right_status,
                pose_landmarks=pose_landmarks,
            )

            vector_valid = (
                len(features)
                == EXPECTED_FEATURE_COUNT
            )

        except ValueError as error:
            features = []
            vector_valid = False
            last_message = str(error)

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        cv2.putText(
            frame,
            "NINJA VISION // FEATURE TEST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"LEFT: {left_status.upper()}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"RIGHT: {right_status.upper()}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        feature_text = (
            f"FEATURES: {len(features)}/"
            f"{EXPECTED_FEATURE_COUNT}"
        )

        cv2.putText(
            frame,
            feature_text,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                (0, 255, 0)
                if vector_valid
                else (0, 0, 255)
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                "VECTOR: VALID"
                if vector_valid
                else "VECTOR: INVALID"
            ),
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                (0, 255, 0)
                if vector_valid
                else (0, 0, 255)
            ),
            2,
        )

        cv2.putText(
            frame,
            "[SPACE] INSPECT VECTOR    [Q] QUIT",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "Ninja Vision - Feature Test",
            frame,
        )

        # -------------------------------------------------
        # Keyboard
        # -------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):

            print()
            print("--------------------------------")
            print("FEATURE VECTOR INSPECTION")
            print("--------------------------------")
            print(
                "Left status:",
                left_status,
            )
            print(
                "Right status:",
                right_status,
            )
            print(
                "Feature count:",
                len(features),
            )

            if features:

                minimum = min(features)
                maximum = max(features)

                average = (
                    sum(features)
                    / len(features)
                )

                zero_count = sum(
                    abs(value) < 1e-8
                    for value in features
                )

                print(
                    f"Minimum: {minimum:.4f}"
                )

                print(
                    f"Maximum: {maximum:.4f}"
                )

                print(
                    f"Average: {average:.4f}"
                )

                print(
                    "Zero features:",
                    zero_count,
                )

                print()
                print(
                    "First 10 features:"
                )

                print(
                    [
                        round(value, 4)
                        for value
                        in features[:10]
                    ]
                )

            print("--------------------------------")

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
    print("Feature test finished.")


if __name__ == "__main__":
    main()
    
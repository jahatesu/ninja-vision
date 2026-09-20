import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.hand_tracker import HandTracker


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)


# ---------------------------------------------------------
# Hand connections
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Draw hand
# ---------------------------------------------------------

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    # Convert normalized MediaPipe coordinates
    # into screen pixel coordinates.
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    # Draw landmarks.
    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1,
        )


# ---------------------------------------------------------
# Status color
# ---------------------------------------------------------

def get_status_color(status):
    """
    Return an OpenCV BGR color depending on tracking state.
    """

    if status == "visible":
        return (0, 255, 0)

    if status == "occluded":
        return (0, 255, 255)

    return (0, 0, 255)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not MODEL_PATH.exists():
        print("ERROR: Hand Landmarker model not found.")
        print(f"Expected location: {MODEL_PATH}")
        return

    # -----------------------------------------------------
    # MediaPipe configuration
    # -----------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = (
        vision.HandLandmarker.create_from_options(
            options
        )
    )

    # -----------------------------------------------------
    # Occlusion tracker
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
        landmarker.close()
        return

    print()
    print("==============================")
    print("       NINJA VISION")
    print("      OCCLUSION TEST")
    print("==============================")
    print()
    print("Try:")
    print("1. Show both hands separately.")
    print("2. Slowly bring them together.")
    print("3. Overlap your palms.")
    print("4. Hold the hand sign.")
    print("5. Separate your hands again.")
    print()
    print("Press Q to exit.")
    print()

    start_time = time.perf_counter()

    # -----------------------------------------------------
    # Main camera loop
    # -----------------------------------------------------

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Mirror webcam.
        frame = cv2.flip(
            frame,
            1
        )

        # OpenCV uses BGR.
        # MediaPipe uses RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert OpenCV frame into MediaPipe image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # VIDEO mode requires increasing timestamps.
        timestamp_ms = int(
            (
                time.perf_counter()
                - start_time
            )
            * 1000
        )

        # Detect hands.
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------------------------
        # Organize currently detected hands
        # -------------------------------------------------

        detected_hands = {
            "left": None,
            "right": None,
        }

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            # Make sure handedness information exists.
            if hand_index >= len(result.handedness):
                continue

            classification = (
                result.handedness[hand_index][0]
            )

            hand_name = (
                classification
                .category_name
                .lower()
            )

            if hand_name == "left":
                detected_hands["left"] = hand_landmarks

            elif hand_name == "right":
                detected_hands["right"] = hand_landmarks

        # -------------------------------------------------
        # Update temporal hand tracker
        # -------------------------------------------------

        tracked_hands = hand_tracker.update(
            detected_hands
        )

        # -------------------------------------------------
        # Draw ONLY currently visible hands
        # -------------------------------------------------

        for hand_name in ("left", "right"):

            hand_data = tracked_hands[hand_name]

            if hand_data["status"] == "visible":
                draw_hand(
                    frame,
                    hand_data["landmarks"]
                )

        # -------------------------------------------------
        # Tracking statuses
        # -------------------------------------------------

        left_status = (
            tracked_hands["left"]["status"]
        )

        right_status = (
            tracked_hands["right"]["status"]
        )

        # -------------------------------------------------
        # Determine overall state
        # -------------------------------------------------

        visible_count = sum(
            status == "visible"
            for status in (
                left_status,
                right_status,
            )
        )

        occluded_count = sum(
            status == "occluded"
            for status in (
                left_status,
                right_status,
            )
        )

        if visible_count == 2:
            overall_status = "BOTH HANDS TRACKED"

        elif (
            visible_count == 1
            and occluded_count == 1
        ):
            overall_status = "TEMPORARY OCCLUSION"

        elif visible_count == 1:
            overall_status = "ONE HAND DETECTED"

        else:
            overall_status = "SEARCHING FOR HANDS"

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        cv2.putText(
            frame,
            "NINJA VISION // OCCLUSION TEST",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Left hand status
        cv2.putText(
            frame,
            f"LEFT: {left_status.upper()}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            get_status_color(left_status),
            2,
        )

        # Right hand status
        cv2.putText(
            frame,
            f"RIGHT: {right_status.upper()}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            get_status_color(right_status),
            2,
        )

        # Overall tracking state
        cv2.putText(
            frame,
            overall_status,
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # Legend
        cv2.putText(
            frame,
            "GREEN=VISIBLE  YELLOW=OCCLUDED  RED=MISSING",
            (20, frame.shape[0] - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            "[Q] QUIT",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        cv2.imshow(
            "Ninja Vision - Occlusion Test",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    camera.release()
    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print("Ninja Vision occlusion test stopped.")


if __name__ == "__main__":
    main()
from utils.landmark_utils import normalize_hand_landmarks
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# ---------------------------------------------------------
# Hand connections
# ---------------------------------------------------------

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
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
# Drawing
# ---------------------------------------------------------

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    # Convert normalized coordinates into pixels.
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    # Draw connections first.
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
        )

    # Draw landmark points.
    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 255, 0),
            -1,
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        print("ERROR: Hand Landmarker model not found.")
        print(f"Expected location: {MODEL_PATH}")
        return

    # Create MediaPipe model configuration.
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

    landmarker = vision.HandLandmarker.create_from_options(
        options
    )

    # Open webcam.
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not access webcam.")
        landmarker.close()
        return

    print("Ninja Vision landmark inspector started.")
    print("Show one or both hands to the camera.")
    print("Press Q to exit.")

    start_time = time.perf_counter()

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Mirror webcam.
        frame = cv2.flip(frame, 1)

        # OpenCV = BGR
        # MediaPipe = RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy image to MediaPipe Image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # VIDEO mode requires monotonically increasing timestamps.
        timestamp_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hand_count = len(result.hand_landmarks)

        # -------------------------------------------------
        # Process detected hands
        # -------------------------------------------------

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            # Draw hand skeleton.
            draw_hand(frame, hand_landmarks)

            # ---------------------------------------------
            # Determine handedness FIRST
            # ---------------------------------------------

            hand_name = "Unknown"
            confidence = 0.0

            if hand_index < len(result.handedness):
                classification = result.handedness[hand_index][0]

                hand_name = classification.category_name
                confidence = classification.score

            # ---------------------------------------------
            # Important landmarks
            # ---------------------------------------------

            wrist = hand_landmarks[0]
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]
            features = normalize_hand_landmarks(
                hand_landmarks
            )

            # ---------------------------------------------
            # Print index fingertip coordinates
            # ---------------------------------------------

            print(
                f"{hand_name} | "
                f"Features: {len(features)} | "
                f"First 6: "
                f"{[round(value, 3) for value in features[:6]]}"
            )

            # ---------------------------------------------
            # Display handedness on webcam
            # ---------------------------------------------

            y_position = 110 + (hand_index * 35)

            cv2.putText(
                frame,
                f"{hand_name.upper()} HAND: {confidence:.0%}",
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        cv2.putText(
            frame,
            "NINJA VISION",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"HANDS DETECTED: {hand_count}/2",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "Ninja Vision - Landmark Inspector",
            frame
        )

        # Q = Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    landmarker.close()
    cv2.destroyAllWindows()

    print("Ninja Vision landmark inspector stopped.")


if __name__ == "__main__":
    main()
import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.landmark_utils import create_two_hand_features


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hand_landmarker.task"
)

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


# ---------------------------------------------------------
# Supported gestures
# ---------------------------------------------------------

SUPPORTED_GESTURES = {
    "tiger",
    "ram",
    "snake",
    "shadow_clone",
    "none",
}


# ---------------------------------------------------------
# Hand connections
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Drawing
# ---------------------------------------------------------

def draw_hand(frame, landmarks):
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

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
            5,
            (0, 255, 0),
            -1,
        )


# ---------------------------------------------------------
# Save sample
# ---------------------------------------------------------

def save_sample(
    file_path,
    label,
    features,
):
    file_exists = file_path.exists()

    with file_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            header = ["label"]

            header += [
                f"feature_{index}"
                for index in range(len(features))
            ]

            writer.writerow(header)

        writer.writerow(
            [label] + features
        )


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Collect hand gesture samples "
            "for Ninja Vision."
        )
    )

    parser.add_argument(
        "--label",
        required=True,
        choices=sorted(SUPPORTED_GESTURES),
        help="Gesture label to collect.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    args = parse_arguments()

    label = args.label

    if not MODEL_PATH.exists():
        print(
            "ERROR: Hand Landmarker model "
            "was not found."
        )
        print(MODEL_PATH)
        return

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        DATA_DIR
        / f"{label}.csv"
    )

    # -----------------------------------------------------
    # MediaPipe
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
    # Camera
    # -----------------------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not access webcam.")
        landmarker.close()
        return

    sample_count = 0
    start_time = time.perf_counter()

    print()
    print("==============================")
    print("       NINJA VISION")
    print("      DATA COLLECTOR")
    print("==============================")
    print()
    print(f"Gesture: {label.upper()}")
    print()
    print("SPACE = Capture sample")
    print("Q     = Quit")
    print()

    while True:
        success, frame = camera.read()

        if not success:
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

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # -------------------------------------------------
        # Organize left/right hands
        # -------------------------------------------------

        left_hand = None
        right_hand = None

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            draw_hand(
                frame,
                hand_landmarks,
            )

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
                left_hand = hand_landmarks

            elif hand_name == "right":
                right_hand = hand_landmarks

        # -------------------------------------------------
        # HUD
        # -------------------------------------------------

        detected_count = len(
            result.hand_landmarks
        )

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
            0.7,
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

        cv2.putText(
            frame,
            f"HANDS: {detected_count}/2",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "[SPACE] CAPTURE    [Q] QUIT",
            (20, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "Ninja Vision - Dataset Collector",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        # -------------------------------------------------
        # Capture
        # -------------------------------------------------

        if key == ord(" "):
            if (
                left_hand is None
                and right_hand is None
            ):
                print(
                    "No hands detected. "
                    "Sample not captured."
                )

                continue

            features = create_two_hand_features(
                left_hand=left_hand,
                right_hand=right_hand,
            )

            if len(features) != 131:
                print(
                    "ERROR: Unexpected "
                    "feature count:",
                    len(features),
                )

                continue

            save_sample(
                output_file,
                label,
                features,
            )

            sample_count += 1

            print(
                f"Captured {label}: "
                f"{sample_count}"
            )

        elif key == ord("q"):
            break

    camera.release()
    landmarker.close()

    cv2.destroyAllWindows()

    print()
    print(
        f"Collection finished. "
        f"{sample_count} samples captured."
    )
    print(
        f"Saved to: {output_file}"
    )


if __name__ == "__main__":
    main()
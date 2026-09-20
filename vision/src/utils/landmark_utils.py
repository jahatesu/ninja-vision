import math


LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3


def normalize_hand_landmarks(landmarks):
    """
    Convert MediaPipe hand landmarks into a normalized feature vector.

    Normalization:
    1. Use the wrist as the origin.
    2. Translate every landmark relative to the wrist.
    3. Scale the coordinates using the maximum 3D distance
       from the wrist.

    Returns:
        list[float]: 63 normalized values
                     (21 landmarks × x, y, z)
    """

    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(
            f"Expected {LANDMARK_COUNT} landmarks, "
            f"received {len(landmarks)}."
        )

    # Landmark 0 = wrist
    wrist = landmarks[0]

    relative_landmarks = []

    # -----------------------------------------------------
    # Translation normalization
    # -----------------------------------------------------

    for landmark in landmarks:
        relative_x = landmark.x - wrist.x
        relative_y = landmark.y - wrist.y
        relative_z = landmark.z - wrist.z

        relative_landmarks.append(
            (
                relative_x,
                relative_y,
                relative_z,
            )
        )

    # -----------------------------------------------------
    # Scale normalization
    # -----------------------------------------------------

    max_distance = max(
        math.sqrt(
            x ** 2 +
            y ** 2 +
            z ** 2
        )
        for x, y, z in relative_landmarks
    )

    # Avoid division by zero.
    if max_distance == 0:
        max_distance = 1.0

    # -----------------------------------------------------
    # Flatten into ML feature vector
    # -----------------------------------------------------

    features = []

    for x, y, z in relative_landmarks:
        features.extend(
            [
                x / max_distance,
                y / max_distance,
                z / max_distance,
            ]
        )

    return features
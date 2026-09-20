import math


# =========================================================
# Constants
# =========================================================

LANDMARK_COUNT = 21

FEATURES_PER_HAND = 63

# Important fingertip landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGERTIP_INDICES = [
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
]


# =========================================================
# Basic math
# =========================================================

def distance_3d(point_a, point_b):
    """
    Calculate Euclidean distance between two MediaPipe
    landmarks.
    """

    return math.sqrt(
        (point_a.x - point_b.x) ** 2
        + (point_a.y - point_b.y) ** 2
        + (point_a.z - point_b.z) ** 2
    )


# =========================================================
# Single-hand normalization
# =========================================================

def normalize_hand_landmarks(landmarks):
    """
    Convert 21 MediaPipe hand landmarks into 63 normalized
    features.

    Normalization:
    - Wrist becomes the origin.
    - Coordinates are translated relative to the wrist.
    - Coordinates are scale-normalized.

    Returns:
        63 floats
    """

    if landmarks is None:
        return [0.0] * FEATURES_PER_HAND

    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(
            f"Expected {LANDMARK_COUNT} landmarks, "
            f"received {len(landmarks)}."
        )

    wrist = landmarks[0]

    relative_landmarks = []

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
    # Determine hand scale
    # -----------------------------------------------------

    max_distance = max(
        math.sqrt(
            x ** 2
            + y ** 2
            + z ** 2
        )
        for x, y, z in relative_landmarks
    )

    if max_distance < 1e-8:
        max_distance = 1.0

    # -----------------------------------------------------
    # Flatten
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


# =========================================================
# Hand scale
# =========================================================

def get_hand_scale(landmarks):
    """
    Estimate hand size.

    Uses wrist-to-middle-fingertip distance.
    """

    if landmarks is None:
        return 1.0

    scale = distance_3d(
        landmarks[0],
        landmarks[MIDDLE_TIP],
    )

    if scale < 1e-8:
        return 1.0

    return scale


# =========================================================
# Inter-hand features
# =========================================================

def calculate_inter_hand_features(
    left_hand,
    right_hand,
):
    """
    Describe the spatial relationship between the two hands.

    Returns 18 features:

    3  = normalized right-wrist position relative to left
    5  = matching fingertip distances
    10 = cross-fingertip distances
    """

    # If either hand is unavailable, the relationship
    # cannot be measured.
    if left_hand is None or right_hand is None:
        return [0.0] * 18

    left_scale = get_hand_scale(left_hand)
    right_scale = get_hand_scale(right_hand)

    average_scale = (
        left_scale + right_scale
    ) / 2.0

    if average_scale < 1e-8:
        average_scale = 1.0

    features = []

    # -----------------------------------------------------
    # Wrist-to-wrist relationship
    # -----------------------------------------------------

    left_wrist = left_hand[0]
    right_wrist = right_hand[0]

    features.extend(
        [
            (
                right_wrist.x
                - left_wrist.x
            ) / average_scale,

            (
                right_wrist.y
                - left_wrist.y
            ) / average_scale,

            (
                right_wrist.z
                - left_wrist.z
            ) / average_scale,
        ]
    )

    # -----------------------------------------------------
    # Matching fingertip distances
    #
    # left thumb  <-> right thumb
    # left index  <-> right index
    # etc.
    # -----------------------------------------------------

    for index in FINGERTIP_INDICES:

        distance = distance_3d(
            left_hand[index],
            right_hand[index],
        )

        features.append(
            distance / average_scale
        )

    # -----------------------------------------------------
    # Cross-fingertip relationships
    #
    # Each left fingertip compared with the neighboring
    # important fingertips on the right hand.
    # -----------------------------------------------------

    cross_pairs = [
        (THUMB_TIP, INDEX_TIP),
        (INDEX_TIP, THUMB_TIP),

        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, INDEX_TIP),

        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, MIDDLE_TIP),

        (RING_TIP, PINKY_TIP),
        (PINKY_TIP, RING_TIP),

        (THUMB_TIP, PINKY_TIP),
        (PINKY_TIP, THUMB_TIP),
    ]

    for left_index, right_index in cross_pairs:

        distance = distance_3d(
            left_hand[left_index],
            right_hand[right_index],
        )

        features.append(
            distance / average_scale
        )

    return features


# =========================================================
# Visibility features
# =========================================================

def create_visibility_features(
    left_status,
    right_status,
):
    """
    Convert tracking states into numerical features.

    Each hand gets three values:

    visible
    occluded
    missing

    Total = 6 features.
    """

    valid_states = {
        "visible",
        "occluded",
        "missing",
    }

    if left_status not in valid_states:
        raise ValueError(
            f"Invalid left hand status: {left_status}"
        )

    if right_status not in valid_states:
        raise ValueError(
            f"Invalid right hand status: {right_status}"
        )

    features = []

    for status in (
        left_status,
        right_status,
    ):

        features.extend(
            [
                1.0 if status == "visible" else 0.0,
                1.0 if status == "occluded" else 0.0,
                1.0 if status == "missing" else 0.0,
            ]
        )

    return features


# =========================================================
# Pose features
# =========================================================

def create_pose_features(
    pose_landmarks=None,
):
    """
    Extract arm geometry from MediaPipe Pose.

    Uses:
        left shoulder  = 11
        right shoulder = 12
        left elbow     = 13
        right elbow    = 14
        left wrist     = 15
        right wrist    = 16

    Coordinates are normalized relative to the midpoint
    between the shoulders.

    Returns:
        18 coordinate features
        +
        6 visibility features

        = 24 features
    """

    if pose_landmarks is None:
        return [0.0] * 24

    required_indices = [
        11,
        12,
        13,
        14,
        15,
        16,
    ]

    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]

    center_x = (
        left_shoulder.x
        + right_shoulder.x
    ) / 2.0

    center_y = (
        left_shoulder.y
        + right_shoulder.y
    ) / 2.0

    center_z = (
        left_shoulder.z
        + right_shoulder.z
    ) / 2.0

    shoulder_width = distance_3d(
        left_shoulder,
        right_shoulder,
    )

    if shoulder_width < 1e-8:
        shoulder_width = 1.0

    features = []

    # -----------------------------------------------------
    # Normalized coordinates
    # -----------------------------------------------------

    for index in required_indices:

        landmark = pose_landmarks[index]

        features.extend(
            [
                (
                    landmark.x
                    - center_x
                ) / shoulder_width,

                (
                    landmark.y
                    - center_y
                ) / shoulder_width,

                (
                    landmark.z
                    - center_z
                ) / shoulder_width,
            ]
        )

    # -----------------------------------------------------
    # Visibility
    # -----------------------------------------------------

    for index in required_indices:

        landmark = pose_landmarks[index]

        visibility = getattr(
            landmark,
            "visibility",
            0.0,
        )

        features.append(
            float(visibility)
        )

    return features


# =========================================================
# Final Ninja Vision feature vector
# =========================================================

def create_gesture_features(
    left_hand=None,
    right_hand=None,
    left_status="missing",
    right_status="missing",
    pose_landmarks=None,
):
    """
    Create the complete Ninja Vision feature vector.

    Layout:

    Left hand landmarks:
        63

    Right hand landmarks:
        63

    Hand visibility:
        6

    Inter-hand relationships:
        18

    Pose / arm features:
        24

    ----------------------------

    TOTAL:
        174 features
    """

    # -----------------------------------------------------
    # Hand geometry
    # -----------------------------------------------------

    left_features = normalize_hand_landmarks(
        left_hand
    )

    right_features = normalize_hand_landmarks(
        right_hand
    )

    # -----------------------------------------------------
    # Visibility
    # -----------------------------------------------------

    visibility_features = (
        create_visibility_features(
            left_status,
            right_status,
        )
    )

    # -----------------------------------------------------
    # Hand relationships
    # -----------------------------------------------------

    relationship_features = (
        calculate_inter_hand_features(
            left_hand,
            right_hand,
        )
    )

    # -----------------------------------------------------
    # Pose
    # -----------------------------------------------------

    pose_features = create_pose_features(
        pose_landmarks
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    features = (
        left_features
        + right_features
        + visibility_features
        + relationship_features
        + pose_features
    )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    expected_feature_count = 174

    if len(features) != expected_feature_count:
        raise ValueError(
            "Unexpected feature vector size. "
            f"Expected {expected_feature_count}, "
            f"received {len(features)}."
        )

    return features
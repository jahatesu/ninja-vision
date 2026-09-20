import math


# =========================================================
# Constants
# =========================================================

LANDMARK_COUNT = 21
FEATURES_PER_HAND = 63

WRIST = 0

# Thumb
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

# Index
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

# Middle
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

# Ring
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

# Pinky
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


FINGERTIP_INDICES = [
    THUMB_TIP,
    INDEX_TIP,
    MIDDLE_TIP,
    RING_TIP,
    PINKY_TIP,
]


# =========================================================
# Basic vector math
# =========================================================

def vector_between(point_a, point_b):
    """
    Return vector A -> B.
    """

    return (
        point_b.x - point_a.x,
        point_b.y - point_a.y,
        point_b.z - point_a.z,
    )


def vector_length(vector):
    return math.sqrt(
        vector[0] ** 2
        + vector[1] ** 2
        + vector[2] ** 2
    )


def distance_3d(point_a, point_b):
    """
    Euclidean distance between two MediaPipe landmarks.
    """

    return vector_length(
        vector_between(
            point_a,
            point_b,
        )
    )


def dot_product(vector_a, vector_b):
    return (
        vector_a[0] * vector_b[0]
        + vector_a[1] * vector_b[1]
        + vector_a[2] * vector_b[2]
    )


def cross_product(vector_a, vector_b):
    return (
        vector_a[1] * vector_b[2]
        - vector_a[2] * vector_b[1],

        vector_a[2] * vector_b[0]
        - vector_a[0] * vector_b[2],

        vector_a[0] * vector_b[1]
        - vector_a[1] * vector_b[0],
    )


def normalize_vector(vector):
    length = vector_length(vector)

    if length < 1e-8:
        return (0.0, 0.0, 0.0)

    return (
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    )


def angle_between_vectors(
    vector_a,
    vector_b,
):
    """
    Return the angle between two vectors normalized to 0–1.

    0.0 = 0 degrees
    0.5 = 90 degrees
    1.0 = 180 degrees
    """

    length_a = vector_length(vector_a)
    length_b = vector_length(vector_b)

    if (
        length_a < 1e-8
        or length_b < 1e-8
    ):
        return 0.0

    cosine = (
        dot_product(
            vector_a,
            vector_b,
        )
        / (length_a * length_b)
    )

    # Floating-point protection.
    cosine = max(
        -1.0,
        min(1.0, cosine),
    )

    angle = math.acos(cosine)

    return angle / math.pi


def joint_angle(
    point_a,
    point_b,
    point_c,
):
    """
    Calculate the angle A-B-C.

    B is the joint being measured.

    Returns normalized angle 0–1.
    """

    vector_ba = vector_between(
        point_b,
        point_a,
    )

    vector_bc = vector_between(
        point_b,
        point_c,
    )

    return angle_between_vectors(
        vector_ba,
        vector_bc,
    )


# =========================================================
# Hand normalization
# =========================================================

def normalize_hand_landmarks(landmarks):
    """
    Convert one hand into 63 normalized coordinates.

    The wrist becomes the origin and the hand is normalized
    by its maximum landmark distance from the wrist.
    """

    if landmarks is None:
        return [0.0] * FEATURES_PER_HAND

    if len(landmarks) != LANDMARK_COUNT:
        raise ValueError(
            f"Expected {LANDMARK_COUNT} landmarks, "
            f"received {len(landmarks)}."
        )

    wrist = landmarks[WRIST]

    relative_landmarks = []

    for landmark in landmarks:

        relative_landmarks.append(
            (
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z,
            )
        )

    max_distance = max(
        vector_length(point)
        for point in relative_landmarks
    )

    if max_distance < 1e-8:
        max_distance = 1.0

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
    Estimate hand scale using wrist -> middle MCP.

    Using the MCP instead of the fingertip makes the scale
    less affected by whether the middle finger is bent.
    """

    if landmarks is None:
        return 1.0

    scale = distance_3d(
        landmarks[WRIST],
        landmarks[MIDDLE_MCP],
    )

    if scale < 1e-8:
        return 1.0

    return scale


# =========================================================
# Finger curl
# =========================================================

def calculate_finger_curl_features(
    landmarks,
):
    """
    Describe how curled each finger is.

    Four measurements are produced per finger:

        MCP angle
        PIP angle
        DIP/IP angle
        fingertip-to-wrist distance

    5 fingers × 4 = 20 features per hand.

    Two hands = 40 features.

    Missing hand = zeros.
    """

    if landmarks is None:
        return [0.0] * 20

    hand_scale = get_hand_scale(
        landmarks
    )

    finger_chains = [
        # Thumb
        (
            THUMB_CMC,
            THUMB_MCP,
            THUMB_IP,
            THUMB_TIP,
        ),

        # Index
        (
            INDEX_MCP,
            INDEX_PIP,
            INDEX_DIP,
            INDEX_TIP,
        ),

        # Middle
        (
            MIDDLE_MCP,
            MIDDLE_PIP,
            MIDDLE_DIP,
            MIDDLE_TIP,
        ),

        # Ring
        (
            RING_MCP,
            RING_PIP,
            RING_DIP,
            RING_TIP,
        ),

        # Pinky
        (
            PINKY_MCP,
            PINKY_PIP,
            PINKY_DIP,
            PINKY_TIP,
        ),
    ]

    features = []

    for (
        base,
        joint_1,
        joint_2,
        tip,
    ) in finger_chains:

        # Angle at first joint.
        angle_1 = joint_angle(
            landmarks[base],
            landmarks[joint_1],
            landmarks[joint_2],
        )

        # Angle at second joint.
        angle_2 = joint_angle(
            landmarks[joint_1],
            landmarks[joint_2],
            landmarks[tip],
        )

        # Overall finger bend.
        overall_angle = joint_angle(
            landmarks[WRIST],
            landmarks[base],
            landmarks[tip],
        )

        # Tip distance from wrist.
        tip_distance = (
            distance_3d(
                landmarks[WRIST],
                landmarks[tip],
            )
            / hand_scale
        )

        features.extend(
            [
                angle_1,
                angle_2,
                overall_angle,
                tip_distance,
            ]
        )

    return features


# =========================================================
# Additional joint-angle features
# =========================================================

def calculate_joint_angle_features(
    landmarks,
):
    """
    Additional structural finger angles.

    8 features per hand.
    16 for both hands.
    """

    if landmarks is None:
        return [0.0] * 8

    features = [
        # Thumb relationships
        joint_angle(
            landmarks[WRIST],
            landmarks[THUMB_CMC],
            landmarks[THUMB_MCP],
        ),

        joint_angle(
            landmarks[THUMB_CMC],
            landmarks[THUMB_MCP],
            landmarks[THUMB_TIP],
        ),

        # Index
        joint_angle(
            landmarks[WRIST],
            landmarks[INDEX_MCP],
            landmarks[INDEX_TIP],
        ),

        # Middle
        joint_angle(
            landmarks[WRIST],
            landmarks[MIDDLE_MCP],
            landmarks[MIDDLE_TIP],
        ),

        # Ring
        joint_angle(
            landmarks[WRIST],
            landmarks[RING_MCP],
            landmarks[RING_TIP],
        ),

        # Pinky
        joint_angle(
            landmarks[WRIST],
            landmarks[PINKY_MCP],
            landmarks[PINKY_TIP],
        ),

        # Index-middle spread
        angle_between_vectors(
            vector_between(
                landmarks[WRIST],
                landmarks[INDEX_TIP],
            ),
            vector_between(
                landmarks[WRIST],
                landmarks[MIDDLE_TIP],
            ),
        ),

        # Ring-pinky spread
        angle_between_vectors(
            vector_between(
                landmarks[WRIST],
                landmarks[RING_TIP],
            ),
            vector_between(
                landmarks[WRIST],
                landmarks[PINKY_TIP],
            ),
        ),
    ]

    return features


# =========================================================
# Within-hand fingertip distances
# =========================================================

def calculate_fingertip_distance_features(
    landmarks,
):
    """
    Measure distances between neighboring fingertips.

    5 per hand.
    10 total.
    """

    if landmarks is None:
        return [0.0] * 5

    scale = get_hand_scale(
        landmarks
    )

    pairs = [
        (THUMB_TIP, INDEX_TIP),
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
        (THUMB_TIP, PINKY_TIP),
    ]

    return [
        distance_3d(
            landmarks[first],
            landmarks[second],
        ) / scale
        for first, second in pairs
    ]


# =========================================================
# Palm orientation
# =========================================================

def calculate_palm_normal(
    landmarks,
):
    """
    Estimate palm normal using:

        wrist -> index MCP
        wrist -> pinky MCP

    Returns a unit vector.
    """

    if landmarks is None:
        return (
            0.0,
            0.0,
            0.0,
        )

    wrist_to_index = vector_between(
        landmarks[WRIST],
        landmarks[INDEX_MCP],
    )

    wrist_to_pinky = vector_between(
        landmarks[WRIST],
        landmarks[PINKY_MCP],
    )

    normal = cross_product(
        wrist_to_index,
        wrist_to_pinky,
    )

    return normalize_vector(
        normal
    )


def calculate_palm_features(
    left_hand,
    right_hand,
):
    """
    Describe palm orientation.

    Returns 6 features:

    left palm normal XYZ
    +
    right palm normal XYZ
    """

    left_normal = calculate_palm_normal(
        left_hand
    )

    right_normal = calculate_palm_normal(
        right_hand
    )

    return [
        *left_normal,
        *right_normal,
    ]


# =========================================================
# Inter-hand geometry
# =========================================================

def calculate_inter_hand_features(
    left_hand,
    right_hand,
):
    """
    Describe relationships between the two hands.

    Returns 18 features:

    3  wrist relative position
    5  matching fingertip distances
    10 cross-fingertip distances
    """

    if (
        left_hand is None
        or right_hand is None
    ):
        return [0.0] * 18

    left_scale = get_hand_scale(
        left_hand
    )

    right_scale = get_hand_scale(
        right_hand
    )

    average_scale = (
        left_scale + right_scale
    ) / 2.0

    if average_scale < 1e-8:
        average_scale = 1.0

    features = []

    # -----------------------------------------------------
    # Wrist relationship
    # -----------------------------------------------------

    left_wrist = left_hand[WRIST]
    right_wrist = right_hand[WRIST]

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
    # Matching fingertips
    # -----------------------------------------------------

    for index in FINGERTIP_INDICES:

        features.append(
            distance_3d(
                left_hand[index],
                right_hand[index],
            ) / average_scale
        )

    # -----------------------------------------------------
    # Cross-fingertip relationships
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

        features.append(
            distance_3d(
                left_hand[left_index],
                right_hand[right_index],
            ) / average_scale
        )

    return features


# =========================================================
# Visibility
# =========================================================

def create_visibility_features(
    left_status,
    right_status,
):
    """
    One-hot encode:

        visible
        occluded
        missing

    3 values per hand.
    6 total.
    """

    valid_states = {
        "visible",
        "occluded",
        "missing",
    }

    if left_status not in valid_states:
        raise ValueError(
            f"Invalid left status: {left_status}"
        )

    if right_status not in valid_states:
        raise ValueError(
            f"Invalid right status: {right_status}"
        )

    features = []

    for status in (
        left_status,
        right_status,
    ):

        features.extend(
            [
                1.0
                if status == "visible"
                else 0.0,

                1.0
                if status == "occluded"
                else 0.0,

                1.0
                if status == "missing"
                else 0.0,
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
    Extract:

        left shoulder
        right shoulder
        left elbow
        right elbow
        left wrist
        right wrist

    18 normalized XYZ coordinates
    +
    6 visibility values

    = 24 features.
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

    left_shoulder = (
        pose_landmarks[11]
    )

    right_shoulder = (
        pose_landmarks[12]
    )

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

    # Coordinates
    for index in required_indices:

        landmark = (
            pose_landmarks[index]
        )

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

    # Visibility
    for index in required_indices:

        visibility = getattr(
            pose_landmarks[index],
            "visibility",
            0.0,
        )

        features.append(
            float(visibility)
        )

    return features


# =========================================================
# Complete Ninja Vision Feature Vector V2
# =========================================================

def create_gesture_features(
    left_hand=None,
    right_hand=None,
    left_status="missing",
    right_status="missing",
    pose_landmarks=None,
):
    """
    NINJA VISION FEATURE VECTOR V2

    -------------------------------------------------------
    BASE FEATURES
    -------------------------------------------------------

    Left normalized landmarks      63
    Right normalized landmarks     63

    Visibility                     6
    Inter-hand geometry           18
    Pose                           24

    -------------------------------------------------------
    ADVANCED FINGER GEOMETRY
    -------------------------------------------------------

    Finger curl:
        left                       20
        right                      20

    Joint angles:
        left                        8
        right                       8

    Fingertip distances:
        left                        5
        right                       5

    Palm orientation               6

    -------------------------------------------------------

    TOTAL                         246

    NOTE:
    This intentionally favors a richer feature representation
    before the real dataset is collected.
    """

    features = []

    # -----------------------------------------------------
    # Original hand coordinates
    # -----------------------------------------------------

    features.extend(
        normalize_hand_landmarks(
            left_hand
        )
    )

    features.extend(
        normalize_hand_landmarks(
            right_hand
        )
    )

    # -----------------------------------------------------
    # Tracking visibility
    # -----------------------------------------------------

    features.extend(
        create_visibility_features(
            left_status,
            right_status,
        )
    )

    # -----------------------------------------------------
    # Two-hand relationships
    # -----------------------------------------------------

    features.extend(
        calculate_inter_hand_features(
            left_hand,
            right_hand,
        )
    )

    # -----------------------------------------------------
    # Pose
    # -----------------------------------------------------

    features.extend(
        create_pose_features(
            pose_landmarks
        )
    )

    # -----------------------------------------------------
    # Finger curl
    # -----------------------------------------------------

    features.extend(
        calculate_finger_curl_features(
            left_hand
        )
    )

    features.extend(
        calculate_finger_curl_features(
            right_hand
        )
    )

    # -----------------------------------------------------
    # Joint angles
    # -----------------------------------------------------

    features.extend(
        calculate_joint_angle_features(
            left_hand
        )
    )

    features.extend(
        calculate_joint_angle_features(
            right_hand
        )
    )

    # -----------------------------------------------------
    # Fingertip distances
    # -----------------------------------------------------

    features.extend(
        calculate_fingertip_distance_features(
            left_hand
        )
    )

    features.extend(
        calculate_fingertip_distance_features(
            right_hand
        )
    )

    # -----------------------------------------------------
    # Palm orientation
    # -----------------------------------------------------

    features.extend(
        calculate_palm_features(
            left_hand,
            right_hand,
        )
    )

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    expected_feature_count = 246

    if len(features) != expected_feature_count:
        raise ValueError(
            "Feature Vector V2 size mismatch. "
            f"Expected {expected_feature_count}, "
            f"received {len(features)}."
        )

    return features
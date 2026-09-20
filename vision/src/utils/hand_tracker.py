import time


class HandTracker:
    """
    Maintains short-term information about hands when MediaPipe
    temporarily loses one because the hands overlap.

    A hand can have three states:

    visible  = MediaPipe currently detects it
    occluded = MediaPipe lost it recently, but it was just visible
    missing  = The hand has been undetected longer than the grace period
    """

    def __init__(self, grace_period=0.35):
        self.grace_period = grace_period

        self.left_hand = None
        self.right_hand = None

        self.left_last_seen = None
        self.right_last_seen = None

    def update(self, detected_hands):
        """
        Update tracking information.

        detected_hands should look like:

        {
            "left": landmarks or None,
            "right": landmarks or None,
        }

        Returns tracking information for both hands.
        """

        now = time.perf_counter()

        current_left = detected_hands.get("left")
        current_right = detected_hands.get("right")

        # -------------------------------------------------
        # Left hand
        # -------------------------------------------------

        if current_left is not None:
            self.left_hand = current_left
            self.left_last_seen = now
            left_status = "visible"

        elif (
            self.left_hand is not None
            and self.left_last_seen is not None
            and now - self.left_last_seen <= self.grace_period
        ):
            left_status = "occluded"

        else:
            self.left_hand = None
            left_status = "missing"

        # -------------------------------------------------
        # Right hand
        # -------------------------------------------------

        if current_right is not None:
            self.right_hand = current_right
            self.right_last_seen = now
            right_status = "visible"

        elif (
            self.right_hand is not None
            and self.right_last_seen is not None
            and now - self.right_last_seen <= self.grace_period
        ):
            right_status = "occluded"

        else:
            self.right_hand = None
            right_status = "missing"

        # -------------------------------------------------
        # Return tracking state
        # -------------------------------------------------

        return {
            "left": {
                "landmarks": self.left_hand,
                "status": left_status,
            },
            "right": {
                "landmarks": self.right_hand,
                "status": right_status,
            },
        }
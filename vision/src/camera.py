import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not access the webcam.")
        return

    print("Ninja Vision camera started.")
    print("Press Q to exit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        # Mirror the camera so movement feels natural.
        frame = cv2.flip(frame, 1)

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
            "CAMERA ACTIVE",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Ninja Vision - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

    print("Ninja Vision camera stopped.")


if __name__ == "__main__":
    main()
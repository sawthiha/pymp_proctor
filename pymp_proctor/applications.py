import cv2

from .proctor import Proctor


def draw_text_center(image, text, origin, font, scale, color, thickness):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    offset_x, offset_y = int(text_size[0] / 2), int(text_size[1] / 2)
    return cv2.putText(
        image,
        text,
        (origin[0] - offset_x, origin[1] + offset_y),
        font,
        scale,
        color,
        thickness,
    )


def draw_stats(img, width, height, idx, result):
    y = 50 + idx * 300
    img = cv2.rectangle(
        img, (width - 350, y), (width - 50, y + 220), (128, 128, 128), -1
    )
    img = draw_text_center(
        img,
        f"Person {idx}",
        (width - 200, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    img = draw_text_center(
        img,
        "Up",
        (width - 200, y + 89),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if result["vertical_label"] == "up" else (0, 0, 0),
        2,
    )
    if result["left_blink"]:
        img = draw_text_center(
            img,
            "Blink",
            (width - 300, y + 89),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    if result["right_blink"]:
        img = draw_text_center(
            img,
            "Blink",
            (width - 100, y + 89),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
    img = draw_text_center(
        img,
        "Left",
        (width - 300, y + 133),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if result["horizontal_label"] == "left" else (0, 0, 0),
        2,
    )
    img = draw_text_center(
        img,
        "Right",
        (width - 100, y + 133),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if result["horizontal_label"] == "right" else (0, 0, 0),
        2,
    )
    img = draw_text_center(
        img,
        "Down",
        (width - 200, y + 177),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if result["vertical_label"] == "down" else (0, 0, 0),
        2,
    )
    return img


def webcam_proctor():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("MediaPipe Proctoring Toolkit", cv2.WINDOW_NORMAL)
    with Proctor(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as proctor:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = proctor.proctor(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for idx, result in enumerate(results):
                image = draw_stats(image, image.shape[1], image.shape[0], idx, result)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow("MediaPipe Proctoring Toolkit", image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

# Copyright 2020 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe Face Mesh."""


from typing import NamedTuple, Tuple

import cv2
import numpy as np

# pylint: disable=unused-import
from mediapipe.calculators.core import (
    constant_side_packet_calculator_pb2,
    gate_calculator_pb2,
    split_vector_calculator_pb2,
)
from mediapipe.calculators.tensor import (
    image_to_tensor_calculator_pb2,
    inference_calculator_pb2,
    tensors_to_classification_calculator_pb2,
    tensors_to_detections_calculator_pb2,
    tensors_to_landmarks_calculator_pb2,
)
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import (
    association_calculator_pb2,
    detections_to_rects_calculator_pb2,
    landmarks_refinement_calculator_pb2,
    logic_calculator_pb2,
    non_max_suppression_calculator_pb2,
    rect_transformation_calculator_pb2,
    thresholding_calculator_pb2,
)

# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase

# pylint: disable=unused-import
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_IRISES,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_TESSELATION,
)

# pylint: enable=unused-import

FACEMESH_NUM_LANDMARKS = 468
FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478
_BINARYPB_FILE_PATH = "mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb"


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def get_distance(p1, p2):
    return np.sqrt(np.square(p1 - p2).sum(axis=0))


def eval_left_ear(landmarks):
    vert_1 = get_distance(landmarks[158], landmarks[153])
    vert_2 = get_distance(landmarks[160], landmarks[144])
    hor = get_distance(landmarks[33], landmarks[133])
    return (vert_1 + vert_2) / (2 * hor)


def eval_right_ear(landmarks):
    vert_1 = get_distance(landmarks[387], landmarks[373])
    vert_2 = get_distance(landmarks[385], landmarks[380])
    hor = get_distance(landmarks[362], landmarks[263])
    return (vert_1 + vert_2) / (2 * hor)


def eval_left_eld(landmarks):
    return get_distance(landmarks[159], landmarks[145])


def eval_right_eld(landmarks):
    return get_distance(landmarks[386], landmarks[374])


def eval_face_orientation(landmarks):
    return landmarks[1][0], landmarks[1][1]


def standardize(landmarks):
    landmarks = landmarks.landmark
    xs = np.array([landmark.x for landmark in landmarks])
    ys = np.array([landmark.y for landmark in landmarks])
    zs = np.array([landmark.z for landmark in landmarks])
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    return np.stack([std_xs, std_ys, std_zs], axis=1)


def landmarks_decor(func):
    def wrapper(landmarks, width, height):
        landmarks = landmarks.landmark
        xs = np.array([landmark.x for landmark in landmarks])
        ys = np.array([landmark.y for landmark in landmarks])
        zs = np.array([landmark.z for landmark in landmarks])
        return func(xs, ys, zs, width, height)

    return wrapper


@landmarks_decor
def candid_ear_model(xs, ys, zs, width, height):
    candid_xs = xs * width
    candid_ys = ys * height
    landmarks = np.stack([candid_xs, candid_ys], axis=1)
    leftEAR = eval_left_ear(landmarks)
    rightEAR = eval_right_ear(landmarks)
    ear = (leftEAR + rightEAR) / 2.0
    return (
        int(ear < 0.2),
        int(leftEAR < 0.2),
        int(rightEAR < 0.2),
        ear,
        leftEAR,
        rightEAR,
    )


@landmarks_decor
def adaptive_eld_model(xs, ys, zs, _, __):
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    landmarks = np.stack([std_xs, std_ys], axis=1)
    o_x, o_y = eval_face_orientation(landmarks)

    leftELD = eval_left_eld(landmarks)
    rightELD = eval_right_eld(landmarks)
    eld = (leftELD + rightELD) / 2.0
    threshold = 0.4163 * o_y
    return (
        int(eld < threshold),
        int(leftELD < threshold),
        int(rightELD < threshold),
        eld,
        leftELD,
        rightELD,
    )


@landmarks_decor
def adaptive_ear_model(xs, ys, zs, width, height):
    std_xs = (xs - xs.mean()) / xs.std()
    std_ys = (ys - ys.mean()) / ys.std()
    std_zs = (zs - zs.mean()) / zs.std()
    landmarks = np.stack([std_xs, std_ys], axis=1)
    o_x, o_y = eval_face_orientation(landmarks)

    leftEAR = eval_left_ear(landmarks)
    rightEAR = eval_right_ear(landmarks)
    threshold = (-0.0401 * o_x) + (0.4241 * o_y)
    ear = (leftEAR + rightEAR) / 2.0
    return (
        int(ear < threshold),
        int(leftEAR < threshold),
        int(rightEAR < threshold),
        ear,
        leftEAR,
        rightEAR,
    )


def vertical_alignment(y):
    if y <= 0.058:
        # if y <= -0.05:
        return "up"
    elif y >= 0.6:
        return "down"
    else:
        return "straight"


def horizontal_alignment(x):
    if x <= -0.3:
        return "left"
    elif x >= 0.3:
        return "right"
    else:
        return "straight"


class Proctor(SolutionBase):
    """MediaPipe Proctor.
    MediaPipe Face Mesh processes an RGB image and returns the face landmarks on
    each detected face.
    Please refer to https://solutions.mediapipe.dev/face_mesh#python-solution-api
    for usage examples.
    """

    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        """Initializes a MediaPipe Face Mesh object.
        Args:
          static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/face_mesh#static_image_mode.
          max_num_faces: Maximum number of faces to detect. See details in
            https://solutions.mediapipe.dev/face_mesh#max_num_faces.
          refine_landmarks: Whether to further refine the landmark coordinates
            around the eyes and lips, and output additional landmarks around the
            irises. Default to False. See details in
            https://solutions.mediapipe.dev/face_mesh#refine_landmarks.
          min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
          min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            face landmarks to be considered tracked successfully. See details in
            https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
        """
        super().__init__(
            binary_graph_path=_BINARYPB_FILE_PATH,
            side_inputs={
                "num_faces": max_num_faces,
                "with_attention": refine_landmarks,
                "use_prev_landmarks": not static_image_mode,
            },
            calculator_params={
                "facedetectionshortrangecpu__facedetectionshortrangecommon__TensorsToDetectionsCalculator.min_score_thresh":
                # 'facedetectionshortrangecpu__facedetectionshortrange__facedetection__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
                "facelandmarkcpu__ThresholdingCalculator.threshold":
                # 'facelandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
            },
            outputs=["multi_face_landmarks", "face_rects_from_detections"],
        )

    def process(self, image: np.ndarray) -> NamedTuple:
        """Processes an RGB image and returns the face landmarks on each detected face.
        Args:
          image: An RGB image represented as a numpy ndarray.
        Raises:
          RuntimeError: If the underlying graph throws any error.
          ValueError: If the input image is not three channel RGB.
        Returns:
          A NamedTuple object with a "multi_face_landmarks" field that contains the
          face landmarks on each detected face.
        """

        return super().process(input_data={"image": image})

    prev_landmarks = None
    prev_nose = None

    def _evaluate(self, landmarks):

        cur_nose = np.array(
            [landmarks.landmark[1].x, landmarks.landmark[1].y, landmarks.landmark[1].z]
        )
        delta = (
            np.linalg.norm(cur_nose - self.prev_nose, ord=2)
            if self.prev_nose is not None
            else 0.0
        )
        self.prev_nose = cur_nose

        landmarks = standardize(landmarks)
        o_x, o_y = eval_face_orientation(landmarks)

        leftELD = eval_left_eld(landmarks)
        rightELD = eval_right_eld(landmarks)
        threshold = (
            (-0.0228 * o_x) + (0.0162 * o_y) + (0.0792 * np.exp(np.square(o_y))) + 0.08
        )

        facial_activity = (
            np.linalg.norm(landmarks - self.prev_landmarks, ord=2)
            if self.prev_landmarks is not None
            else 0.0
        )
        self.prev_landmarks = landmarks

        return {
            "horizontal": o_x,
            "horizontal_label": horizontal_alignment(o_x),
            "vertical": o_y,
            "vertical_label": vertical_alignment(o_y),
            "left_eld": leftELD,
            "right_eld": rightELD,
            "threshold": threshold,
            "left_blink": leftELD < threshold,
            "right_blink": rightELD < threshold,
            "facial_activity": facial_activity,
            "movement": delta,
        }

    def proctor(self, image: np.ndarray) -> Tuple[dict]:
        results = self.process(image)
        if results.multi_face_landmarks == None:
            return []

        return [self._evaluate(landmarks) for landmarks in results.multi_face_landmarks]

    def face_align(
        self,
        image: np.ndarray,
        desired_interocular_distance=44,
        face_center=(0.5, 0.4),
        size=(224, 224),
    ) -> Tuple[Tuple[np.ndarray], NamedTuple]:
        results = self.process(image)

        aligneds = []
        if (
            results.face_rects_from_detections == None
            or results.multi_face_landmarks == None
        ):
            return aligneds, results
        for detection, landmarks in zip(
            results.face_rects_from_detections, results.multi_face_landmarks
        ):
            # Calculate the actual pixel values for the center offset and size of the crop
            center_offset_pixels = (
                int(detection.y_center * image.shape[0]),
                int(detection.x_center * image.shape[1]),
            )
            size_pixels = (
                int(detection.height * image.shape[0]),
                int(detection.width * image.shape[1]),
            )

            # Calculate the start and end indices for the crop
            start_y = clamp(
                int(center_offset_pixels[0] - size_pixels[0] / 2), 0, image.shape[0]
            )
            end_y = clamp(
                int(center_offset_pixels[0] + size_pixels[0] / 2), 0, image.shape[0]
            )
            start_x = clamp(
                int(center_offset_pixels[1] - size_pixels[1] / 2), 0, image.shape[1]
            )
            end_x = clamp(
                int(center_offset_pixels[1] + size_pixels[1] / 2), 0, image.shape[1]
            )

            # Crop the image using the start and end indices
            cropped_image = image[start_y:end_y, start_x:end_x]

            # Get the left and right eye corner coordinates
            cropped_scale_x, cropped_scale_y = (
                size[1] / cropped_image.shape[1],
                size[0] / cropped_image.shape[0],
            )
            landmarks = (
                np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark])
                * np.array([image.shape[1], image.shape[0]])
                - np.array([start_x, start_y])
            ) * np.array([cropped_scale_x, cropped_scale_y])

            left_eye_x, left_eye_y = landmarks[
                [
                    7,
                    33,
                    33,
                    144,
                    145,
                    153,
                    154,
                    155,
                    157,
                    158,
                    159,
                    160,
                    161,
                    163,
                    173,
                    246,
                ]
            ].mean(axis=0)
            right_eye_x, right_eye_y = landmarks[
                [
                    249,
                    263,
                    263,
                    373,
                    374,
                    380,
                    381,
                    382,
                    384,
                    385,
                    386,
                    387,
                    388,
                    390,
                    398,
                    466,
                ]
            ].mean(axis=0)

            cropped_image = cv2.resize(
                cropped_image, size, interpolation=cv2.INTER_LINEAR
            )

            # Calculate the image center
            image_center_x = cropped_image.shape[1] * face_center[0]
            image_center_y = cropped_image.shape[0] * face_center[1]

            # Calculate the center of the face
            face_center_x = (left_eye_x + right_eye_x) // 2
            face_center_y = (left_eye_y + right_eye_y) // 2

            # Calculate the angle of rotation
            dY = left_eye_y - right_eye_y
            dX = left_eye_x - right_eye_x
            rotation_angle = np.degrees(np.arctan2(dY, dX)) - 180

            # Calculate the interocular distance
            interocular_distance = np.sqrt(
                (right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2
            )

            # Calculate the scaling factor
            scaling_factor = desired_interocular_distance / interocular_distance

            M = cv2.getRotationMatrix2D(
                (face_center_x, face_center_y), rotation_angle, scaling_factor
            )
            M[0, 2] += image_center_x - face_center_x
            M[1, 2] += image_center_y - face_center_y

            # Apply the rotation, translation, and scaling transformations to the image
            aligneds.append(
                cv2.warpAffine(
                    cropped_image,
                    M,
                    (cropped_image.shape[1], cropped_image.shape[0]),
                    flags=cv2.INTER_CUBIC,
                )
            )

        return aligneds, results

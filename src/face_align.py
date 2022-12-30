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

import numpy as np
import cv2

# pylint: disable=unused-import
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import association_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmarks_refinement_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python.solution_base import SolutionBase
# pylint: disable=unused-import
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
# pylint: enable=unused-import

FACEMESH_NUM_LANDMARKS = 468
FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478
_BINARYPB_FILE_PATH = 'mediapipe/modules/face_landmark/face_landmark_front_cpu.binarypb'

def clamp(value, min_value, max_value):
  return max(min(value, max_value), min_value)

class FaceAlign(SolutionBase):
  """MediaPipe Face Mesh.
  MediaPipe Face Mesh processes an RGB image and returns the face landmarks on
  each detected face.
  Please refer to https://solutions.mediapipe.dev/face_mesh#python-solution-api
  for usage examples.
  """

  def __init__(self,
               static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
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
            'num_faces': max_num_faces,
            'with_attention': refine_landmarks,
            'use_prev_landmarks': not static_image_mode,
        },
        calculator_params={
            'facedetectionshortrangecpu__facedetectionshortrange__facedetection__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'facelandmarkcpu__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['multi_face_landmarks', 'face_rects_from_detections'])

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

    return super().process(input_data={'image': image})

  def face_align(self, image: np.ndarray, desired_interocular_distance = 44, face_center=(0.5, 0.4), size=(224, 224)) -> Tuple[Tuple[np.ndarray], NamedTuple]:
    results = self.process(image)
    
    aligneds = []
    if results.face_rects_from_detections == None or results.multi_face_landmarks == None:
      return aligneds, results
    for detection, landmarks in zip(results.face_rects_from_detections, results.multi_face_landmarks):
      # Calculate the actual pixel values for the center offset and size of the crop
      center_offset_pixels = (int(detection.y_center * image.shape[0]), int(detection.x_center * image.shape[1]))
      size_pixels = (int(detection.height * image.shape[0]), int(detection.width * image.shape[1]))

      # Calculate the start and end indices for the crop
      start_y = clamp(int(center_offset_pixels[0] - size_pixels[0] / 2), 0, image.shape[0])
      end_y = clamp(int(center_offset_pixels[0] + size_pixels[0] / 2), 0, image.shape[0])
      start_x = clamp(int(center_offset_pixels[1] - size_pixels[1] / 2), 0, image.shape[1])
      end_x = clamp(int(center_offset_pixels[1] + size_pixels[1] / 2), 0, image.shape[1])

      # Crop the image using the start and end indices
      cropped_image = image[start_y:end_y, start_x:end_x]

      # Get the left and right eye corner coordinates
      cropped_scale_x, cropped_scale_y = size[1] / cropped_image.shape[1], size[0] / cropped_image.shape[0]
      landmarks = (np.array([[landmark.x, landmark.y] for landmark in landmarks.landmark]) * np.array([image.shape[1], image.shape[0]]) - np.array([start_x, start_y])) * np.array([cropped_scale_x, cropped_scale_y])

      left_eye_x, left_eye_y = landmarks[[7, 33, 33, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246]].mean(axis=0)
      right_eye_x, right_eye_y = landmarks[[249, 263, 263, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466]].mean(axis=0)

      cropped_image = cv2.resize(cropped_image, size, interpolation=cv2.INTER_LINEAR)

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
      interocular_distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)

      # Calculate the scaling factor
      scaling_factor = desired_interocular_distance / interocular_distance

      M = cv2.getRotationMatrix2D((face_center_x, face_center_y), rotation_angle, scaling_factor)
      M[0, 2] += image_center_x - face_center_x
      M[1, 2] += image_center_y - face_center_y

      # Apply the rotation, translation, and scaling transformations to the image
      aligneds.append(cv2.warpAffine(cropped_image, M, (cropped_image.shape[1], cropped_image.shape[0]), flags=cv2.INTER_CUBIC))

    return aligneds, results
"""Tests for hello function."""
import cv2
import pytest
from mediapipe.python.solutions.face_mesh import FaceMesh

from pymp_proctor.proctor import Proctor


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "assets/images/test.jpeg",
            [
                {
                    "horizontal_label": "right",
                    "vertical_label": "straight",
                    "left_blink": False,
                    "right_blink": False,
                }
            ],
        ),
    ],
)
def test_hello(src, expected):
    """Example test with parametrization."""
    with Proctor(static_image_mode=True, refine_landmarks=True) as proctor:
        image = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        results = proctor.proctor(image)
        assert results is not None
        assert len(results) == len(expected)
        for result, expect in zip(results, expected):
            for key, value in expect.items():
                assert result[key] == value

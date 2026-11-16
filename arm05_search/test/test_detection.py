#!/usr/bin/env python3
"""
Test detekcji BEZ symulacji: bierze teksturę kostki z arm05_sim, "ustawia" ją wirtualnie
1 m przed kamerą, rzutuje na obraz 1920x1080 i sprawdza, czy Twoje detect_markers()
i estimate_pose() znajdują właściwe ID i odległość.

    cd /arm_ws/src/arm05_search && python3 test/test_detection.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bt_scripts'))
from aruco_detector import ArucoDetector  # noqa: E402

TEXTURES = Path('/arm_ws/src/arm05_sim/models/aruco_markers')
CUBE = 0.09          # bok kostki = bok tekstury [m]
DISTANCE = 1.0       # jak daleko ustawiamy kostkę [m]


class FakeDetector:
    """Udaje node: te same pola, których używają metody ArucoDetector (bez ROS)."""
    marker_size = 0.07
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    detector_params = cv2.aruco.DetectorParameters_create()
    # parametry kamery z symulacji (1920x1080, poziome FOV 1.02974 rad)
    fx = 1920 / 2 / np.tan(1.02974 / 2)
    camera_matrix = np.array([[fx, 0, 960], [0, fx, 540], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(5)
    detect_markers = ArucoDetector.detect_markers
    estimate_pose = ArucoDetector.estimate_pose


def render(texture: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Rzutuje teksturę jako kwadrat CUBE x CUBE ustawiony DISTANCE metrów przed kamerą."""
    h, w = texture.shape[:2]
    half = CUBE / 2
    corners_3d = np.array([[-half, -half, DISTANCE], [half, -half, DISTANCE],
                           [half, half, DISTANCE], [-half, half, DISTANCE]])
    projected = (K @ corners_3d.T).T
    dst = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((1080, 1920, 3), 180, dtype=np.uint8)   # szare tło
    warped = cv2.warpPerspective(texture, H, (1920, 1080))
    mask = cv2.warpPerspective(np.ones((h, w), np.uint8), H, (1920, 1080)).astype(bool)
    canvas[mask] = warped[mask]
    return canvas


def main():
    det = FakeDetector()
    ok = True
    for marker_id in range(5):
        texture = cv2.imread(str(TEXTURES / str(marker_id) / 'materials' / 'textures' / f'{marker_id}.png'))
        frame = render(texture, det.camera_matrix)
        corners, ids = det.detect_markers(frame)
        if ids is None or len(ids) == 0:
            print(f'[FAIL] ID {marker_id}: nic nie wykryto')
            ok = False
            continue
        tvec = det.estimate_pose(corners[0])
        dist = float(np.linalg.norm(tvec))
        good = int(ids[0][0]) == marker_id and abs(dist - DISTANCE) < 0.05
        ok &= good
        print(f'[{"OK" if good else "FAIL"}] oczekiwano ID {marker_id}, wykryto {ids.ravel().tolist()}, '
              f'odległość {dist:.3f} m (oczekiwano {DISTANCE})')
    print('\nWSZYSTKO OK' if ok else '\nSą błędy - popraw i uruchom ponownie')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

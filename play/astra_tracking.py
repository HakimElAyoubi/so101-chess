"""Short-lived, anchored tracking of a feature Astra identified; never selects a new target.

A small immutable image patch prevents cumulative drift. Each frame must have a
strong, unique match nearby. Occlusion, repeated patterns, weak texture, large
motion, frame edges or age invalidate the track and require a new Astra look.
The controller also requires reacquisition after changing grasp height/opening.
"""
import time

import cv2
import numpy as np


class FeatureTracker:
    def __init__(self, gray, pixel):
        self.gray = gray.copy()
        self.pixel = np.asarray(pixel, np.float32)
        self.started, self.steps = time.monotonic(), 0
        self.reason, self.score = "", 1.0
        self.template = cv2.getRectSubPix(gray, (33, 33), tuple(self.pixel))
        features = cv2.goodFeaturesToTrack(self.template, maxCorners=40, qualityLevel=.025,
                                           minDistance=3, blockSize=3)
        self.n_seed = 0 if features is None else len(features)
        self.valid = self.n_seed >= 4 and float(self.template.std()) >= 10
        if not self.valid:
            self.reason = "weak target texture"

    def invalidate(self, reason):
        self.valid, self.reason = False, reason
        return False

    def update(self, gray):
        if not self.valid:
            return False
        if gray.shape != self.gray.shape or time.monotonic() - self.started > 10.0:
            return self.invalidate("track expired")
        h, w = gray.shape
        if not (35 <= self.pixel[0] < w - 35 and 35 <= self.pixel[1] < h - 35):
            return self.invalidate("target at frame edge")
        if np.array_equal(gray, self.gray):
            return True
        # The search is bounded to 18 raw pixels per frame. Keep the original
        # template throughout; updating it every tick would silently learn drift.
        search = cv2.getRectSubPix(gray, (69, 69), tuple(self.pixel))
        scores = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        self.score = float(score)
        if not np.isfinite(score) or score < .80:
            return self.invalidate("target appearance changed")
        x, y = loc
        other = scores.copy()
        other[max(0, y - 5):y + 6, max(0, x - 5):x + 6] = -1
        if score - float(other.max()) < .08:
            return self.invalidate("ambiguous target match")
        # Reject peaks at the search boundary: motion could have exceeded the window.
        if not (0 < x < scores.shape[1] - 1 and 0 < y < scores.shape[0] - 1):
            return self.invalidate("target moved too far")

        def subpixel(a, b, c):
            denominator = float(a - 2 * b + c)
            return float(np.clip(.5 * (a - c) / denominator, -.5, .5)) if abs(denominator) > 1e-6 else 0.0
        dx = subpixel(scores[y, x - 1], scores[y, x], scores[y, x + 1])
        dy = subpixel(scores[y - 1, x], scores[y, x], scores[y + 1, x])
        self.pixel += np.array([x + dx - 18, y + dy - 18], np.float32)
        self.gray = gray.copy()
        self.steps += 1
        return True

"""OpenAI vision backend for the calibrated grasp controller (legacy Astra command).

Default: astra:gpt-6.1-terra (2026-11-10: half of Sol's price, Roboflow Vision Evals 85.8 % vs 81.7 %) at low effort, standard tier; scripted:place for placement.
Use astra:mock for a no-API twin run.
Uses the Responses API over stdlib HTTPS; no additional package installation.
"""
import json
import math
import os
import sys
import time
import threading
import random
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import cv2
import numpy as np
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llm_skill import ClaudeSkill, TOP_NAME, PIECE_H_CM, DESCEND_MM_S, CLOSE_GRIP, TOP_Z_CM, CROP, SCALE, _b64_png, PADS_APPLY_MAX_MM
from astra_tracking import FeatureTracker
from vision_costs import SpendLedger, SESSION_LEDGER


class AstraAPIError(RuntimeError):
    def __init__(self, status, retry_after_s=None, fatal=False):
        self.api_status, self.retry_after_s, self.fatal = status, retry_after_s, fatal
        reason = "account quota/billing or access requires attention" if fatal else "temporarily unavailable; backing off"
        super().__init__(f"OpenAI HTTP {status}: {reason}")


def api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    path = Path.home() / ".openai_api_key"
    if not key and path.is_file():
        key = path.read_text().strip()
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY or create ~/.openai_api_key (one line, chmod 600).")
    return key


# The old prompt asked for piece localisation, four grid corners, a pad-offset calculation,
# obstacle assessment and prose in every look. The board detector handles the grid now.
SYSTEM_PROMPT = """Locate one target in a wrist-camera image of a chess robot. Return only the requested JSON.
Coordinates are image pixels [x,y], origin top left; use the image's full dimensions.
The thin coloured boundary identifies the intended square approximately, not the target pixel. It may be 10 mm off.
Use the piece's shape, colour and printed board letters to distinguish neighbours. Never point at an overlay.
GRASP: report the specified feature's centre (piece TOP; knight BASE). The top identifies the stem axis;
software will lower the jaws to the demonstrated BODY grasp height. Do not choose the visible side, base or shadow
instead of the specified top. If hidden or uncertain, visible=false, target_px=null; do not invent a centre.
BLACK QUEEN: identify the crown axis from the hover; grasp the narrower body below it, not the crown.
The controller lowers this queen's body-grasp target 11 mm. A clear aligned hover (confidence >=0.85, error <=2 mm) authorizes a straight
descent and settled close; no sideways corrections are allowed at body height; do not substitute a neighbouring piece when the queen's crown is hidden by the pads.
BISHOP: identify the centre of the small top ball as the stem axis, not its mitre edge or slit.
A verified hover authorizes descent and settled close at the demonstrated body height; the top may be hidden by pads then.
between_pads: true ONLY when the INTENDED piece's graspable body is physically between both facing contact pads
at the pads' current height, so closing now would grip its body. False when the jaws hover ABOVE the piece,
when only its image projects between them, or identity/body contact-height is uncertain. Do not require a visible top
if its body and the two pads clearly establish this. For PLACE always false. If true, confidence rates this body-grasp
judgment and target identity; target_px may be null if the specified top is hidden. The controller closes immediately
on a confident fresh true judgment instead of demanding perfect pixel alignment or another downward move.
PLACE: report the printed target square's centre, inferring from its real edges if partly covered.
pad_gap_px: midpoint between the two real white pads' FACING surfaces at their contact tips, not their outer edges.
Use null unless both contact tips are clearly visible. Ignore metal jaw geometry; its predicted gap is inaccurate.
clear=false only if the descent corridor actually intersects another piece/jaw, not merely because neighbours exist.
KNIGHT GRASP: the grasper must turn 90 degrees from its normal approach orientation to close ACROSS the narrow neck,
not along the horse's head. The controller performs this quarter-turn above the piece before visual alignment.
The message states whether that turn is complete. When complete, keep it: roll_deg=0; do not undo or repeat the turn.
For a knight still awaiting the turn report roll_deg=90. For all other pieces and for PLACE, roll_deg=0.
confidence is 0..1 for correct target identity and pixel. Perform this direct visual measurement without prose."""

PAIR = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
NULLABLE_PAIR = {"anyOf": [PAIR, {"type": "null"}]}
LOOK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "visible": {"type": "boolean"}, "target_px": NULLABLE_PAIR,
        "pad_gap_px": NULLABLE_PAIR, "clear": {"type": "boolean"}, "between_pads": {"type": "boolean"},
        "roll_deg": {"type": "number", "minimum": -90, "maximum": 90},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["visible", "target_px", "pad_gap_px", "clear", "roll_deg", "confidence", "between_pads"],
}


def validate_answer(ans, crop):
    """Reject malformed/non-finite/out-of-frame measurements before geometry sees them."""
    if not isinstance(ans, dict) or set(ans) != set(LOOK_SCHEMA["required"]):
        raise ValueError("Astra returned an invalid look object")
    for field in ("visible", "clear", "between_pads"):
        if type(ans[field]) is not bool:
            raise ValueError(f"Invalid {field}")

    def number(x):
        return type(x) in (int, float) and math.isfinite(x)

    def pair(p, in_frame=False):
        if not isinstance(p, list) or len(p) != 2 or not all(number(x) for x in p):
            raise ValueError("Invalid pixel pair")
        if in_frame and not (0 <= p[0] < crop.shape[1] and 0 <= p[1] < crop.shape[0]):
            raise ValueError("Pixel outside the wrist crop")

    if ans["target_px"] is not None:
        pair(ans["target_px"], True)
    elif ans["visible"]:
        raise ValueError("Visible target has no pixel")
    if ans["pad_gap_px"] is not None:
        pair(ans["pad_gap_px"], True)
    for field, lo, hi in (("roll_deg", -90, 90), ("confidence", 0, 1)):
        if not number(ans[field]) or not lo <= ans[field] <= hi:
            raise ValueError(f"Invalid {field}")
    return ans


class AstraSkill(ClaudeSkill):
    type = "astra"
    model_default = "gpt-6.1-terra"
    max_looks = 14
    strict_measurements = True

    def __init__(self, checkpoint="final", variant="skills", repo=None):
        super().__init__(checkpoint, variant, repo)
        self.effort = os.environ.get("ASTRA_EFFORT", "low")
        if self.effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("ASTRA_EFFORT must be low, medium, high, xhigh or max")
        self.service_tier = os.environ.get("ASTRA_SERVICE_TIER", "default")
        if self.service_tier not in ("default", "fast", "priority", "ultrafast"):
            raise ValueError("ASTRA_SERVICE_TIER must be default, fast, priority or ultrafast")
        # Two semantic looks + local corrections; a slow API must not expire just before closing.
        self.timeout_s = 180.0
        self._key = None if self.mock else api_key()
        self._cost_ledger = SpendLedger() if self.mock else SESSION_LEDGER

    def reset(self):
        super().reset()
        self._track_lock = getattr(self, "_track_lock", threading.RLock())
        with self._track_lock:
            self._tracker = None
            self._track_answer = None
            self._track_phase = None
            self._local_looks = 0
            self._knight_turned = False
            self._verified_descent = None
            self._measured_q6 = None
            self._queen_rechecks = 0
            self._fresh_rechecks = 0
            self._api_errors = 0
            self._api_wait_until = 0.
            self._api_deadline = None
            self.fatal_error = None

    def _knight_turn_target(self, q6):
        """Choose an equivalent +/-90 degree wrist turn, preserving tip position and tilt within joint limits."""
        q6 = np.asarray(q6, float)
        target, _, _ = self.tips(q6)
        candidates = []
        for delta in (-90.0, 90.0):
            seed = q6.copy(); seed[4] += delta
            model_q = self.proj.C.lerobot_to_qpos(seed)[:5]
            if not self.Pl.K.lo[4] <= model_q[4] <= self.Pl.K.hi[4]:
                continue
            candidate = self.joints_for_tips(seed, target, iters=30, tol=.002)
            if candidate is None:
                continue
            model_q = self.proj.C.lerobot_to_qpos(candidate)[:5]
            if np.all(model_q >= self.Pl.K.lo) and np.all(model_q <= self.Pl.K.hi):
                candidates.append(candidate)
        return min(candidates, key=lambda q: np.abs(q[:5] - q6[:5]).max()) if candidates else None

    def _within_joint_limits(self, q6):
        if q6 is None:
            return False
        model_q = self.proj.C.lerobot_to_qpos(q6)[:5]
        return bool(np.all(model_q >= self.Pl.K.lo) and np.all(model_q <= self.Pl.K.hi))

    def _orient_knight(self, q6):
        """Return a command while raising/turning, or None when normal visual control can continue."""
        if self.phase != "settle_look" or self.low or not self.raised or self._knight_turned or self.move_t0 is not None:
            return None
        if self._targets()[2] != "knight":
            return None
        now = time.monotonic()
        if not self._settled(q6, now):
            return None
        tips, _, _ = self.tips(q6)
        # The knight's pointed feature is its BASE, but turning clearance must use its physical TOP.
        safe_z = float(self._targets()[0][2]) + PIECE_H_CM["knight"] / 100 + .020
        if tips[2] < safe_z - .003:
            target = tips.copy(); target[2] = safe_z
            new = self.joints_for_tips(q6, target, iters=30, tol=.002)
            if not self._within_joint_limits(new) or self._knight_turn_target(new) is None:
                # The near-rank wrist may already be at its flex limit. Use the
                # planner's bounded tilt fallback instead of pushing through it.
                site, _ = self.Pl.fk(q6); site[2] += safe_z - tips[2]
                tilt, psi = self.Pl.orientation(q6)
                seed, _ = self.Pl.ik_near(site, tilt, psi, q6, self.grip)
                new = self.joints_for_tips(seed, target, iters=30, tol=.002) if seed is not None else None
                if not self._within_joint_limits(new) or self._knight_turn_target(new) is None:
                    new = None
            if new is None:
                self.phase, self.gave_up = "give_up", "no IK for knight turning clearance"
            else:
                self._begin_move(new, now, max(.7, (safe_z - tips[2]) * 1000 / DESCEND_MM_S))
            return self._command(q6)
        new = self._knight_turn_target(q6)
        if new is None:
            self.phase, self.gave_up = "give_up", "no reachable 90-degree knight grasp orientation"
            return self._command(q6)
        # Same 30 deg/s min-jerk peak as the staged paths; no rapid wrist snap.
        duration = max(.7, 1.875 * float(np.abs(new[:5] - self.desired[:5]).max()) / 30)
        self._begin_move(new, now, duration)
        self._knight_turned = True
        print(f"[astra skill] knight quarter-turn: {new[4] - q6[4]:+.0f} degrees before visual alignment", file=sys.stderr, flush=True)
        with self._track_lock:
            self._tracker = None
        return self._command(q6)

    def act(self, state6, image_hwc_uint8, task):
        self._measured_q6 = np.asarray(state6, float).copy()
        if self.fatal_error:
            return self._command(state6)
        if self.phase == "api_backoff":
            now = time.monotonic()
            if now < self._api_wait_until:
                return self._command(state6)
            self.phase = "settle_look_low" if self.low else "settle_look"
            self.settle_start, self.settle_t0 = now, None
        if self._verified_descent is not None and self.phase == "settle_look_low" and self.move_t0 is None:
            return self._finish_verified_descent(np.asarray(state6, float), image_hwc_uint8)
        if self.desired is not None:
            command = self._orient_knight(np.asarray(state6, float))
            if command is not None:
                return command
        # This is local image work only. No network or FK runs on the 30 Hz control thread.
        if not self.mock:
            gray = cv2.cvtColor(image_hwc_uint8, cv2.COLOR_RGB2GRAY)
            with self._track_lock:
                if self._tracker is not None:
                    self._tracker.update(gray)
        return super().act(state6, image_hwc_uint8, task)

    def grasp_height_cm(self, piece):
        height = super().grasp_height_cm(piece)
        if piece == "queen" and getattr(self.staged, "colour", None) == "black":
            height -= 1.10
        return height

    def _down(self, kind, now):
        self._verified_descent = None
        piece = self._targets()[2]
        if kind == "from" and self._uses_verified_grasp():
            answer = self.answer or {}
            if (answer.get("visible") and answer.get("clear") and answer.get("confidence", 0) >= .85 and
                    self.last_dist is not None and self.last_dist <= 2.0):
                # A clipped servo command can differ from the measured hover
                # (recorded wrist: desired 106 deg, actual 102 deg). Rebase the
                # descent on the physical pose and discard the old error integral.
                measured = self._measured_q6
                if measured is not None:
                    self.desired = measured.copy()
                    self.bias = np.zeros(6)
                self._verified_descent = dict(piece=piece, xy=self.tips(self.desired)[0][:2].copy(),
                                           confidence=answer["confidence"], hover_error_mm=self.last_dist,
                                           started=now)
            else:
                # Never fall back to chasing an occluded crown at body height.
                self.gave_up = f"{piece} hover not verified for straight descent and close"
                return
        low = super()._down(kind, now)
        if low is False:                                   # the mid-descent look: a fresh model look from there
            self._verified_descent = None                  # re-verified from the mid look before the final descent
            with self._track_lock:
                self._tracker = None
            return False
        if self._verified_descent is not None:
            # Preserve the verified hover axis at the commanded open grip.
            target = self.tips(self.desired)[0].copy()
            target[:2] = self._verified_descent["xy"]
            refined = self.desired.copy(); refined[5] = self.grip
            site_z = self.z_board + self.grasp_height_cm(piece) / 100
            for _ in range(3):
                target[2] = site_z - self.site_above_tips(refined)
                refined = self.joints_for_tips(refined, target, iters=40, tol=.0005)
                if refined is None:
                    break
            if refined is None or not self._within_joint_limits(refined):
                self._verified_descent = None
                self.move_t0 = None
                self.gave_up = f"cannot preserve {piece} alignment at lower body height"
                print(f"[astra skill] {piece} descent rejected: {self.gave_up}", file=sys.stderr, flush=True)
                return
            self._begin_move(refined, now, self.move_T)
            self._verified_descent["deadline"] = now + self.move_T + 5.0
            with self._track_lock:
                self._tracker = None
        return low

    def _forget_tracker(self):
        with self._track_lock:
            self._tracker = None

    def visual_grasp_ready(self, ans, meta, q6, kind):
        if (kind != "from" or ans.get("_source") != "astra" or
                ans.get("between_pads") is not True or not ans.get("clear") or
                ans.get("confidence", 0) < .85 or ans.get("pad_gap_px") is None):
            return False
        # A cached tracker cannot certify 3D contact. The arm must have stayed
        # where the fresh image was taken while the model answered.
        if np.max(np.abs(np.asarray(meta["q"])[:5] - np.asarray(q6)[:5])) > 1.5:
            return False
        self._verified_descent = None
        return True

    def handle_look_error(self, pending, now, kind):
        if pending.get("api_status") != 429 and not pending.get("api_fatal"):
            return False
        self.look_n -= 1  # transport errors are not visual measurements
        self.desired = self._measured_q6.copy(); self.bias = np.zeros(6)
        self._api_errors += 1
        if self._api_deadline is None:
            self._api_deadline = now + 60.
        server_delay = pending.get("api_retry_after") or 0.
        delay = max(server_delay, 5. * 2 ** (self._api_errors - 1)) + random.uniform(0, 1)
        if (pending.get("api_fatal") or self._api_errors > 3 or now + delay > self._api_deadline or
                now + delay > self.t_phase + self.timeout_s):
            self.fatal_error = ("Local estimated API spending cap reached" if pending.get("api_status") == "budget" else
                                "Astra API quota/billing/access requires attention" if pending.get("api_fatal") else
                                "Astra API 429 persists; stopped after bounded backoff")
            self.phase = "api_unavailable"
            return True
        self._api_wait_until = now + delay
        self.phase = "api_backoff"
        with self._track_lock:
            self._tracker = None
        print(f"[astra skill] HTTP 429: hold and wait {delay:.1f}s before a fresh camera look", file=sys.stderr, flush=True)
        return True

    def _uses_verified_grasp(self):
        return (getattr(self.staged, "piece", None) == "bishop" and
                "from" in (self.goal or {}).get("which", ("from",))) or self._black_queen_grasp()

    def _black_queen_grasp(self):
        return (getattr(self.staged, "piece", None) == "queen" and
                getattr(self.staged, "colour", None) == "black" and
                "from" in (self.goal or {}).get("which", ("from",)))

    def review_action(self, action, why, ans, meta, kind, dist):
        if (kind == "from" and not self.low and action == "descend" and ans.get("_source") == "local_tracker"
                and self._fresh_rechecks < (2 if getattr(self, "mid_done", False) else 1) and self.look_n < self.max_looks):
            # The local tracker said 'converged' after the hover correction, yet the first e7 descents landed beside the
            # pawn (real arm, 2026-11-10; the retry, verified by a fresh model look, landed on it). One fresh model look
            # before committing to the descent; a second lateral correction is still accepted from the tracker.
            self._fresh_rechecks += 1
            with self._track_lock:
                self._tracker = None
            return "relook", "fresh look before the descent"
        if kind != "from" or self.low or not self._uses_verified_grasp():
            return action, why
        visible = ans.get("visible") and ans.get("target_px") is not None
        confidence = ans.get("confidence", 0) or 0
        if visible and ans.get("clear") and dist is not None:
            if action == "descend" and confidence < .85 and not ans.get("_agreement"):   # two agreeing looks count as confirmed
                if self._queen_rechecks < 2 and self.look_n < self.max_looks:
                    self._queen_rechecks += 1
                    with self._track_lock:
                        self._tracker = None  # require a fresh model observation, not the same cached confidence
                    return "relook", "confirm hover confidence before descent"
                return "give_up", "grasp hover confidence not confirmed after rechecks"
            if (self._black_queen_grasp() and action == "give_up" and confidence >= .85 and 2.0 < dist <= 6.0 and
                    4 <= self.moves_hover < 6 and self.look_n < self.max_looks):
                return "move", "bounded extra queen hover correction"
        return action, why

    def _command(self, q6):
        cmd = super()._command(q6)
        if not self.mock and self._uses_verified_grasp():
            model_q = self.proj.C.lerobot_to_qpos(cmd)
            model_q[:5] = np.clip(model_q[:5], self.Pl.K.lo, self.Pl.K.hi)
            bounded = self.proj.C.qpos_to_lerobot(model_q)
            # Anti-windup: do not integrate an error beyond the joint limit.
            self.bias[:5] += bounded[:5] - cmd[:5]
            cmd[:5] = bounded[:5]
        return cmd

    def _lateral(self, q_meas, dx, dy, now, roll_deg=0.0):
        if self.low and self._uses_verified_grasp():
            self.gave_up = f"no lateral {self._targets()[2]} corrections at body height"
            return 0.0, 0.0
        if not self._uses_verified_grasp():                # bishop and queen: the wrist runs out of travel on the back rank
            return super()._lateral(q_meas, dx, dy, now, roll_deg)
        piece = self._targets()[2]
        if self.mock:
            return super()._lateral(q_meas, dx, dy, now, roll_deg)
        measured = self._measured_q6 if self._measured_q6 is not None else np.asarray(q_meas, float)
        tips_now = self.tips(q_meas)[0]
        # The elevated camera pose can run out of wrist travel. Lower only
        # within a verified above-crown hover envelope, preserving tilt and roll.
        clearance_z = float(self._targets()[0][2]) + max(PIECE_H_CM.get(piece, TOP_Z_CM[piece]), TOP_Z_CM[piece]) / 100 + .015
        for fraction in (1., .5, .25):
            for drop in (0., .005, .010, .015):
                target = tips_now + np.array([dx * fraction / 1000, dy * fraction / 1000, -drop])
                if target[2] < clearance_z:
                    continue
                candidate = self.joints_for_tips(measured, target, iters=40, tol=.0005)
                if not self._within_joint_limits(candidate):
                    continue
                if any(self.tips(measured + u * (candidate - measured))[0][2] < clearance_z - .001
                       for u in np.linspace(0, 1, 11)):
                    continue
                self.desired = measured.copy(); self.bias = np.zeros(6)
                duration = max(.7, 1.875 * float(np.abs(candidate[:5] - measured[:5]).max()) / 30.,
                               1000 * np.linalg.norm(target - self.tips(measured)[0]) / DESCEND_MM_S)
                self._begin_move(candidate, now, duration)
                with self._track_lock:
                    self._tracker = None  # remeasure the real pad gap after height/pose changes
                if drop:
                    print(f"[astra skill] {piece} hover: {drop * 1000:.0f} mm lower to keep correction within joint limits",
                          file=sys.stderr, flush=True)
                return dx * fraction, dy * fraction
        self.gave_up = f"no reachable {self._targets()[2]} hover correction within joint limits"
        return 0., 0.

    def _finish_verified_descent(self, q6, img_rgb):
        """Close on the verified piece axis after a straight descent.

        The head can be occluded at body height. Do not make new lateral guesses
        there; verify the intended XY and lower site height before commanding contact.
        """
        now = time.monotonic()
        cert = self._verified_descent
        piece = cert["piece"]
        site, _ = self.Pl.fk(q6)
        tips, _, _ = self.tips(q6)
        expected_z = self.z_board + self.grasp_height_cm(piece) / 100
        z_error = 1000 * (float(site[2]) - expected_z)
        xy_error = 1000 * float(np.linalg.norm(tips[:2] - cert["xy"]))
        if self._settled(q6, now) and abs(z_error) <= 1.5 and xy_error <= 2.0:
            self.grip, self.phase, self.t_phase = CLOSE_GRIP, "closed", now
            crop, meta = self.annotate(q6, img_rgb, "from")
            self._log_look("from", dict(phase="low", action="close", method=f"verified {piece} hover + settled vertical descent",
                                       q=q6.tolist(), piece=piece, target_height_cm=self.grasp_height_cm(piece),
                                       height_error_mm=z_error, xy_error_mm=xy_error,
                                       hover_error_mm=cert["hover_error_mm"], confidence=cert["confidence"],
                                       answer={"_source": "verified_descent"}), crop)
            print(f"[astra skill] {piece} body close: site {100 * (site[2] - self.z_board):.2f} cm, "
                  f"height error {z_error:+.1f} mm, XY drift {xy_error:.1f} mm", file=sys.stderr, flush=True)
            self.anchor(q6, meta)
            self._verified_descent = None
        elif now >= cert["deadline"]:
            self.phase, self.gave_up = "give_up", f"{piece} descent did not settle: height {z_error:+.1f} mm, XY drift {xy_error:.1f} mm"
            self._verified_descent = None
        return self._command(q6)

    def register(self, ans, meta):
        # A historical grasp anchor or one/two cropped corners cannot disprove
        # the current target identity or translate the board model reliably.
        ans["_grid_trusted"] = False
        pts = ans.get("grid_px") or []
        if len(pts) < 3 or cv2.contourArea(cv2.convexHull(np.asarray(pts, np.float32))) < 100:
            return {"grid_n": len(pts), "grid_used": 0, "reason": "insufficient 2D grid support"}
        reg = self._reg()
        previous = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in reg.items()}
        info = super().register(ans, meta)
        if info.get("grid_used", 0) < 3:
            reg.clear(); reg.update(previous)
            info.update(grid_used=0, reason="insufficient matched grid support")
        else:
            ans["_grid_trusted"] = True
        return info

    def square_identity_trusted(self, ans):
        return bool(ans.get("_grid_trusted"))

    def annotate(self, q6, img_rgb, kind):
        # Reuse the calibrated crop/metadata, replace the cluttered drawing. No scale grid,
        # goal dot, arrow, PADS reference or text is drawn across the actual target.
        _, meta = super().annotate(q6, img_rgb, kind)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        crop = img[meta["y0"]:meta["y0"] + CROP, meta["x0"]:meta["x0"] + CROP].copy()
        crop = cv2.resize(crop, (int(CROP * SCALE), int(CROP * SCALE)), interpolation=cv2.INTER_CUBIC)
        sq = meta.get("square")
        if sq:
            f, r = "abcdefgh".index(sq[0]), int(sq[1])
            z = meta["z_plane"] - meta["z_board"]
            colour = (0, 190, 0) if kind == "from" else (210, 0, 210)
            # The PRINTED square (board plane) is the unambiguous one: the lifted (top-height) outline is displaced by
            # parallax, a whole square at the back rank where the camera looks obliquely (f8 bishop, 2026-11-10: the lifted
            # outline sat on e8's king and the model answered 'target None'). Printed square solid, lifted one thin.
            for zz, thick in ((0.0, 2), (z, 1)):
                xyz = [self.bp(f + df, r + dr, zz) for df, dr in ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5))]
                uv = (self.project(q6, np.array(xyz))[0] - [meta["x0"], meta["y0"]]) * SCALE
                if np.isfinite(uv).all():
                    cv2.polylines(crop, [np.round(uv).astype(np.int32)], True, colour, thick, cv2.LINE_AA)
            if kind == "from" and getattr(self, "hint_ring", False):
                # recovery look after a 'target None': a thin ring (about 12 mm) around the printed square's centre, with the
                # text telling the model to point at the piece itself. A dot made the model echo the belief; a ring is a search area.
                c_w = self.bp(f, r, 0.0); e_w = np.asarray(c_w, float) + [0.012, 0.0, 0.0]
                uvc = (self.project(q6, np.array([c_w, e_w]))[0] - [meta["x0"], meta["y0"]]) * SCALE
                if np.isfinite(uvc).all():
                    cv2.circle(crop, tuple(np.round(uvc[0]).astype(int)), max(6, int(np.linalg.norm(uvc[1] - uvc[0]))), (0, 190, 0), 1, cv2.LINE_AA)
        return crop, meta

    def _situation_text(self, kind, meta):
        frm, to = self.squares()
        piece = self.staged.piece
        target = (f"GRASP {self.staged.colour} {piece} on {frm}; feature: {TOP_NAME[piece]}."
                  if kind == "from" else f"PLACE held {piece} on printed square {to}.")
        orientation = (" KNIGHT: required 90-degree quarter-turn is complete; maintain this orientation, roll_deg=0."
                       if self._knight_turned else " KNIGHT: rotate grasper 90 degrees before descending, roll_deg=90.") if kind == "from" and piece == "knight" else ""
        return (f"{target} Image {int(CROP * SCALE)} x {int(CROP * SCALE)} pixels. "
                f"Phase: {'at BODY grasp/release height' if self.low else 'hover before descent'}. "
                "The thick green square is the PRINTED square the piece stands on, drawn on the board surface; the thin one is "
                "that square lifted to the piece's top height. Find the piece STANDING on the thick square and point at its "
                "feature; its top appears displaced from the printed square by perspective." + orientation
                + (" A thin green circle marks the centre of that printed square on the board: the piece's BASE should be near "
                   "it; point at the piece itself, never at the circle's centre." if kind == "from" and getattr(self, "hint_ring", False) else ""))

    def _grid_pixels(self, q6, img_rgb, meta):
        # Use the project's tested X-corner/PnP detector instead of asking Astra to enumerate corners.
        # Require a well-supported, colour-consistent, unshifted fit; otherwise keep no grid update.
        if meta.get("square") is None:
            return []
        from board_vision import BoardVision
        bv = BoardVision(self.proj)
        m = bv.measure(img_rgb, q6, meta["square"], self.session)
        if not m or m.get("why") or m.get("used", 0) < 6 or m.get("colour_conf", 0) < .8 or m.get("shift") != [0, 0]:
            return []
        ab = np.array([c[0] for c in m["corners"]], np.float64)
        px = np.array([c[1] for c in m["corners"]], np.float64)
        H, inliers = cv2.findHomography(ab, bv.to_normalized(px, img_rgb.shape[1], img_rgb.shape[0]), cv2.RANSAC, .002)
        if H is None or inliers is None or inliers.sum() < 6:
            return []
        points = (px[inliers.ravel() > 0] - [meta["x0"], meta["y0"]]) * SCALE
        return [p.tolist() for p in points if 0 <= p[0] < CROP * SCALE and 0 <= p[1] < CROP * SCALE]

    def _ask(self, q6, img_rgb, kind):
        if self.mock:
            return super()._ask(q6, img_rgb, kind)
        generation = self._look_generation
        crop, meta = self.annotate(q6, img_rgb, kind)
        text = self._situation_text(kind, meta)
        self.last_look_img = crop
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        with self._track_lock:
            tracker = self._tracker
            # Always reacquire after the descent: the pads can hide the crown, and
            # the real gap must be measured at the new opening/height.
            local = (tracker is not None and self._track_phase == (kind, self.low)
                     and self._local_looks < 2 and tracker.update(gray))
            if local:
                pixel = (tracker.pixel - [meta["x0"], meta["y0"]]) * SCALE
                local = bool(0 <= pixel[0] < crop.shape[1] and 0 <= pixel[1] < crop.shape[0])
            if local:
                ans = dict(self._track_answer)
                ans["between_pads"] = False  # never reuse an old contact judgment
                ans["target_px"] = ((tracker.pixel - [meta["x0"], meta["y0"]]) * SCALE).tolist()
                if meta.get("pads_px") is not None:
                    ans["pad_gap_px"] = (np.asarray(meta["pads_px"]) + ans["pads_offset_px"]).tolist()
                ans.update(_source="local_tracker", _latency_s=0.0, _track_score=tracker.score,
                           _usage={"in": 0, "out": 0, "cached": 0, "effort": self.effort})
                self._local_looks += 1
            else:
                self._tracker = None
        if not local:
            observed = self._model_answer(text, crop)
            if generation != self._look_generation:
                raise RuntimeError("discarded look from obsolete attempt")
            ans = dict(observed)
            gap = observed["pad_gap_px"]
            ans["pads_offset_px"] = ((np.asarray(gap) - meta["pads_px"]).tolist()
                                     if gap is not None and meta.get("pads_px") is not None else [0, 0])
            ans.update(grid_px=[], notes="", _source="astra")
            if ans["visible"] and ans["confidence"] >= .7 and ans["clear"]:
                raw = self.crop_to_raw(meta, ans["target_px"])
                with self._track_lock:
                    if generation == self._look_generation:
                        self._tracker = FeatureTracker(gray, raw)
                        self._track_answer = dict(ans)
                        self._track_phase = (kind, self.low)
                        self._local_looks = 0
        # Local tracking is for grasp features only; place centres can be occluded
        # and their identity must be checked afresh rather than following the held piece.
        if kind == "to":
            with self._track_lock:
                self._tracker = None
        if kind == "from" and self._knight_turned and self._targets()[2] == "knight":
            ans["roll_deg"] = 0.0
        try:
            ans["grid_px"] = self._grid_pixels(q6, img_rgb, meta)
        except (cv2.error, ValueError) as exc:
            ans["grid_px"] = []
            ans["_grid_error"] = type(exc).__name__
        return ans, meta, crop, text

    def measure(self, ans, meta, kind="from"):
        dx, dy, info = super().measure(ans, meta, kind)
        # Journal: the real jaw gap differs from the twin; calibrate against the
        # visible pads before descent as well, without changing the board or demo heights.
        offset = info.get("pads_offset_mm")
        if kind == "from" and not self.low and not self.mock and offset is not None:
            if ans.get("confidence", 0) >= .7 and np.linalg.norm(offset) <= PADS_APPLY_MAX_MM:   # was 6.0: the real 6.4 mm offset was never applied
                dx, dy = dx - offset[0], dy - offset[1]
                info.update(pads_applied=True, d_mm=[dx, dy])
        return dx, dy, info

    def grasp_capture(self, ans, meta, dx, dy):
        """A close is a contact action, not another demand for sub-calibration pixel precision.

        Keep the 2 mm hover aim; at the demonstrated body height allow up to 4 mm
        along the closing axis, 2 mm across it, and 4 mm overall. Require a fresh,
        confident clear target. No missing look or exhausted budget can authorize it.
        """
        result = {"ok": False, "along_limit_mm": 4.0, "across_limit_mm": 2.0, "radius_limit_mm": 4.0}
        if not self.low or not ans.get("visible") or not ans.get("clear") or ans.get("confidence", 0) < .85:
            return result
        if not np.isfinite([dx, dy]).all() or meta.get("z_board") is None:
            return result
        q = np.asarray(meta["q"], float)
        _, a, b = self.tips(q)
        axis = (np.asarray(b) - a)[:2]
        length = np.linalg.norm(axis)
        if length < 1e-6:
            return result
        axis /= length
        delta = np.array([dx, dy])
        along = float(delta @ axis)
        across = float(delta @ np.array([-axis[1], axis[0]]))
        piece = self._targets()[2]
        site, _ = self.Pl.fk(q)
        height_error = 1000 * (float(site[2]) - float(meta["z_board"]) - self.grasp_height_cm(piece) / 100)
        result.update(along_mm=along, across_mm=across, radius_mm=float(np.linalg.norm(delta)),
                      height_error_mm=height_error)
        result["ok"] = bool(abs(along) <= 4 and abs(across) <= 2 and np.linalg.norm(delta) <= 4 and abs(height_error) <= 3)
        return result

    def _post(self, payload):
        req = Request("https://api.openai.com/v1/responses",
                      data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=35.0) as response:
                return json.load(response)
        except HTTPError as exc:
            # Do not dump request headers or raw bodies into the robot's logs.
            code = None
            if exc.code == 429:
                try:
                    error = json.loads(exc.read(65536)).get("error", {})
                    code = error.get("code") or error.get("type")
                except (ValueError, AttributeError, TypeError):
                    pass
            delay = None
            value = (exc.headers or {}).get("Retry-After")
            if value:
                try:
                    delay = float(value)
                except ValueError:
                    try:
                        delay = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        pass
            if delay is not None and (not math.isfinite(delay) or delay < 0):
                delay = None
            fatal = exc.code != 429 or code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active")
            raise AstraAPIError(exc.code, delay, fatal) from None
        except (URLError, TimeoutError):
            raise RuntimeError("OpenAI request failed or timed out; arm holds while the player retries") from None

    def _model_answer(self, text, crop):
        started = time.monotonic()
        payload = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64," + _b64_png(crop), "detail": "high"},
                {"type": "input_text", "text": text},
            ]}],
            "reasoning": {"effort": self.effort},
            "text": {"verbosity": "low", "format": {"type": "json_schema", "name": "robot_look", "strict": True, "schema": LOOK_SCHEMA}},
            "store": False,
            "service_tier": self.service_tier,
        }
        for budget in (1024,):
            payload["max_output_tokens"] = budget
            self.cost_calls += 1
            ledger = self._cost_ledger
            ticket = ledger.reserve(getattr(self, "staged", None) or self, self.model, self.service_tier, budget)
            try:
                resp = self._post(payload)
            except AstraAPIError:
                ledger.finish(ticket, known_unbilled=True)
                raise
            except Exception:
                ledger.finish(ticket)  # reserve conservatively if billing status is unknown
                raise
            cost = ledger.finish(ticket, resp.get("usage"))
            print(f"[vision cost] {self.model} {self.service_tier}: ${cost['estimated_usd']:.4f}, "
                  f"move ${cost['move_estimated_usd']:.4f}, session ${cost['session_estimated_usd']:.4f}", file=sys.stderr, flush=True)
            content = [c for item in resp.get("output", []) if item.get("type") == "message"
                       for c in item.get("content", [])]
            if any(c.get("type") == "refusal" for c in content):
                raise RuntimeError("Astra refused this look; no measurement used")
            if resp.get("status") != "completed":
                raise RuntimeError(f"Astra look did not complete (status {resp.get('status')})")
            txt = "".join(c.get("text", "") for c in content if c.get("type") == "output_text")
            ans = validate_answer(json.loads(txt), crop)
            usage = resp.get("usage") or {}
            ans["_latency_s"] = round(time.monotonic() - started, 2)
            ans["_usage"] = {"in": usage.get("input_tokens", 0), "out": usage.get("output_tokens", 0),
                             "cached": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0), "effort": self.effort}
            ans["_model"] = resp.get("model", self.model)
            ans["_service_tier"] = resp.get("service_tier", self.service_tier)
            ans["_estimated_usd"] = cost["estimated_usd"]
            return ans

"""Board readers: how the game loop learns the game state ('arm looks at the board').

The loop is   watch pose -> read the board -> infer the opponent's move -> Stockfish -> arm executes -> watch pose.
A reader returns the physical position as {square: symbol} ('P' white pawn, 'k' black king ...), the same format
as play_game.expected_symbols / ChessScene.board_state_symbols, and infer_move() turns the difference between the
logical position and the observed one into the (legal) move that was played.

PrivilegedBoardReader  the digital-twin placeholder: renders the wrist image at the current (watch) pose, so the
                       whole loop is exercised including the camera, but returns the simulator's ground truth
                       (ChessScene.board_state_symbols).  `last_image` keeps the frame it rendered.
VisionBoardReader      the real-robot reader (TODO): detect the game state from the wrist image.  Plan: at the watch
                       pose the board corners project to known pixels (ChessScene.project(board_corners)) -> homography
                       to a top-down 8x8 grid -> per-square occupancy / colour classifier (or piece-type classifier)
                       -> symbols; verify against the logical position (only one legal move can explain the change).
                       read() raises NotImplementedError until it exists; play_game uses --reader vision to select it.
                       With the randomised board pose a single watch pose cannot always see the whole board:
                       render_scan() gives the 2-view board scan (near / far ranks) whose union always does.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import chess
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_sim import config as C


def expected_symbols(board: chess.Board) -> Dict[str, str]:
    return {chess.square_name(sq): pc.symbol() for sq, pc in board.piece_map().items()}


def infer_move(board: chess.Board, observed: Dict[str, str]) -> Optional[chess.Move]:
    """The legal move of `board` whose resulting position matches the observed symbols (None when no legal
    move explains the observation: the pieces were not moved, or moved illegally / knocked away).  Promotions
    are told apart by the promoted piece's symbol."""
    for mv in board.legal_moves:
        b = board.copy(stack=False)
        b.push(mv)
        if expected_symbols(b) == observed:
            return mv
    return None


def render_scan(scene, camera: str = C.WRIST_CAM):
    """Wrist images at the board-scan poses (ChessScene.scan_poses: near ranks / far ranks, together the whole board
    for every randomised board pose).  The arm is posed KINEMATICALLY for each view and restored afterwards (no
    physics step; on the real robot the arm moves through the views).  Returns [(lerobot pose, image)]."""
    import mujoco
    d = scene.data
    q, qv = d.qpos.copy(), d.qvel.copy()
    out = []
    try:
        for v in scene.scan_poses():
            d.qpos[scene.qpos_adr] = C.lerobot_to_qpos(v)
            mujoco.mj_forward(scene.model, d)
            out.append((v, scene.render(camera)))
    finally:
        d.qpos[:] = q
        d.qvel[:] = qv
        mujoco.mj_forward(scene.model, d)
    return out


class BoardReader:
    name = "abstract"

    def read(self, scene) -> Dict[str, str]:
        raise NotImplementedError


class PrivilegedBoardReader(BoardReader):
    """Ground truth from the simulator, after rendering the wrist camera (render=True) as the real reader would."""
    name = "privileged"

    def __init__(self, render: bool = True, camera: str = C.WRIST_CAM):
        self.render = render
        self.camera = camera
        self.last_image: Optional[np.ndarray] = None
        self.n_reads = 0

    def read(self, scene) -> Dict[str, str]:
        if self.render:
            self.last_image = scene.render(self.camera)
        self.n_reads += 1
        return scene.board_state_symbols()


class VisionBoardReader(BoardReader):
    """Game state from the wrist image: not implemented yet (see the module docstring for the plan)."""
    name = "vision"

    def __init__(self, camera: str = C.WRIST_CAM):
        self.camera = camera
        self.last_image: Optional[np.ndarray] = None

    def read(self, scene) -> Dict[str, str]:
        self.last_image = scene.render(self.camera)
        raise NotImplementedError("VisionBoardReader: the wrist-image board reader is not implemented yet; use the privileged "
                                  "reader in the digital twin (play_game.py --reader privileged, the default)")


READERS = {"privileged": PrivilegedBoardReader, "vision": VisionBoardReader}


def make_reader(name: str = "privileged", **kw) -> BoardReader:
    if name not in READERS:
        raise ValueError(f"unknown board reader {name!r}; choose from {sorted(READERS)}")
    return READERS[name](**kw)

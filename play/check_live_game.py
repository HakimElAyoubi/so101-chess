"""Offline live-game tests: legal planning, asynchronous engine and physical transactions."""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_game import LiveGame, Stockfish, transfer_plan


class PlayerStub:
    def __init__(self):
        self.policy = SimpleNamespace(type='smolvla')
        self.grasp_policy = None
        self.model_repo, self.grasp_repo = 'test/combined', None
        self.ckpt, self.grasp_ckpt = '007500', 'final'
        self.phase, self.loading, self.standin = 'idle', '', None
        self.is_ready = True
        self.drop_k = 0
        self.replay_path = 'must be cleared'
        self.started = []
        self.fail_plan = False

    def at_ready(self):
        return self.is_ready

    def q(self):
        return np.zeros(6)

    def start_trial(self):
        self.started.append(self.custom)
        if not self.fail_plan:
            self.phase = 'running'

    def handle(self, command):
        if command == 'go_ready':
            self.is_ready, self.phase = True, 'idle'
        elif command == 'stop':
            self.phase = 'idle'

    def load(self, ckpt):
        self.loading = ckpt


class EngineStub:
    path = 'engine stub'
    def choose(self, board):
        return board.parse_san('e5')
    def close(self):
        pass


class Plans(unittest.TestCase):
    def test_castling_is_two_transfers(self):
        board = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1')
        self.assertEqual(transfer_plan(board, board.parse_san('O-O')),
                         ['move the black king from e8 to g8', 'move the black rook from h8 to f8'])
        self.assertEqual(transfer_plan(board, board.parse_san('O-O-O'))[-1], 'move the black rook from a8 to d8')

    def test_capture_removes_victim_first(self):
        board = chess.Board()
        for move in ['e4', 'd5', 'exd5']:
            board.push_san(move)
        tasks = transfer_plan(board, board.parse_san('Qxd5'))
        self.assertEqual(tasks, ['move the white pawn from d5 to the drop zone', 'move the black queen from d8 to d5'])

    def test_en_passant_removes_actual_pawn_square(self):
        board = chess.Board()
        for move in ['a3', 'e5', 'a4', 'e4', 'd4']:
            board.push_san(move)
        self.assertEqual(transfer_plan(board, board.parse_san('exd3')),
                         ['move the white pawn from d4 to the drop zone', 'move the black pawn from e4 to d3'])

    def test_underpromotion_has_pawn_transfer(self):
        board = chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')
        self.assertEqual(transfer_plan(board, board.parse_san('a1=N')), ['move the black pawn from a2 to a1'])

    def test_illegal_move_rejected(self):
        with self.assertRaises(ValueError):
            transfer_plan(chess.Board(), chess.Move.from_uci('e2e5'))


class Transactions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.p = PlayerStub()
        self.g = LiveGame(self.p, engine=EngineStub(), log_root=self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.g.close)

    def start(self):
        self.g.handle('live_calibrated')
        self.g.handle('live_start')
        self.g.tick(time.monotonic()+2, np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(self.g.status, 'human_turn')

    def black_move(self):
        self.start()
        self.g.handle('live_human:e4')
        self.g.future.result(timeout=2)
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'robot_running')

    def staged_done(self, placed=True):
        self.g.robot_finished(dict(staged_result=['placed' if placed else 'failed', 'test']))

    def test_requires_calibration_models_and_ready(self):
        self.g.handle('live_start')
        self.assertEqual(self.g.status, 'setup')
        self.p.policy.type = 'act'
        self.g.handle('live_calibrated')
        self.g.handle('live_start')
        self.assertEqual(self.g.status, 'setup')
        self.p.policy.type = 'smolvla'
        self.p.is_ready = False
        self.g.handle('live_start')
        self.assertEqual(self.g.status, 'setup')

    def test_legal_human_move_and_live_engine_reply(self):
        self.black_move()
        self.assertEqual(self.p.started, ['move the black pawn from e7 to e5'])
        self.assertIsNone(self.p.replay_path)
        self.assertEqual(self.g.board.peek().uci(), 'e2e4')
        self.staged_done()
        self.assertEqual(self.g.status, 'verify_transfer')
        self.assertEqual(self.g.board.peek().uci(), 'e2e4')
        self.g.handle('live_confirm_transfer')
        self.assertEqual(self.g.board.peek().uci(), 'e7e5')
        self.assertEqual(self.g.status, 'watching')
        self.assertTrue(self.g.log_path.exists())
        self.assertIn('robot_move_verified', self.g.log_path.read_text())

    def test_illegal_human_move_leaves_position_unchanged(self):
        self.start()
        self.g.handle('live_human:e2e5')
        self.assertEqual(self.g.board.fen(), chess.STARTING_FEN)
        self.assertEqual(self.g.status, 'human_turn')

    def test_failed_transfer_never_advances_and_retry_is_explicit(self):
        self.black_move()
        before = self.g.board.fen()
        self.staged_done(False)
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'paused')
        self.assertEqual(self.g.board.fen(), before)
        self.assertEqual(len(self.p.started), 1)
        self.g.handle('live_retry')
        self.g.tick(time.monotonic(), None)
        self.assertEqual(len(self.p.started), 2)
        self.assertEqual(self.g.board.fen(), before)

    def test_capture_keeps_logical_board_until_both_transfers_verified(self):
        self.g.board = chess.Board()
        for move in ['e4', 'd5', 'exd5']:
            self.g.board.push_san(move)
        self.g.pending = self.g.board.parse_san('Qxd5')
        self.g.tasks = transfer_plan(self.g.board, self.g.pending)
        self.g.status = 'robot_wait'
        before = self.g.board.fen()
        self.g.tick(time.monotonic(), None)
        self.staged_done()
        self.g.handle('live_confirm_transfer')
        self.assertEqual(self.g.board.fen(), before)
        self.assertEqual(self.g.drop_used, 1)
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.p.custom, 'move the black queen from d8 to d5')
        self.staged_done()
        self.g.handle('live_confirm_transfer')
        self.assertEqual(self.g.board.piece_at(chess.D5).symbol(), 'q')
        self.assertIsNone(self.g.pending)

    def test_promotion_requires_manual_swap_before_commit(self):
        self.g.board = chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')
        self.g.pending = self.g.board.parse_san('a1=N')
        self.g.tasks = transfer_plan(self.g.board, self.g.pending)
        self.g.status = 'robot_running'
        self.staged_done()
        self.g.handle('live_confirm_transfer')
        self.assertEqual(self.g.status, 'promotion')
        self.assertEqual(self.g.board.piece_at(chess.A2).symbol(), 'p')
        self.g.handle('live_confirm_promotion')
        self.assertEqual(self.g.board.piece_at(chess.A1).symbol(), 'n')

    def test_stop_prevents_next_transfer_and_commit(self):
        self.black_move()
        self.g.handle('live_stop')
        before = self.g.board.fen()
        self.staged_done()
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'stopped')
        self.assertEqual(self.g.board.fen(), before)
        self.assertEqual(len(self.p.started), 1)

    def test_model_choice_supports_combined_and_separate_checkpoints(self):
        self.g.handle('live_models:{"repo":"owner/combined","checkpoint":"007500"}')
        self.assertEqual(self.p.model_repo, 'owner/combined')
        self.assertIsNone(self.p.grasp_repo)
        self.assertEqual(self.p.loading, '007500')
        self.p.loading = ''
        self.g.handle('live_models:{"repo":"owner/place","grasp_repo":"owner/grasp","grasp_checkpoint":"010000"}')
        self.assertEqual(self.p.grasp_repo, 'owner/grasp')
        self.assertEqual(self.p.grasp_ckpt, '010000')

    def test_model_change_blocked_during_pending_move(self):
        self.black_move()
        self.g.handle('live_models:{"repo":"owner/other"}')
        self.assertEqual(self.p.model_repo, 'test/combined')

    def test_api_or_invalid_checkpoint_is_not_accepted(self):
        for args in ['{"repo":"astra:model"}', '{"repo":"owner/model","checkpoint":"../bad"}']:
            self.g.handle('live_models:'+args)
            self.assertEqual(self.p.model_repo, 'test/combined')

    def test_planning_failure_stays_pending(self):
        self.p.fail_plan = True
        self.start()
        self.g.handle('live_human:e4')
        self.g.future.result(timeout=2)
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'paused')
        self.assertEqual(self.g.pending.uci(), 'e7e5')
        self.assertEqual(self.g.board.peek().uci(), 'e2e4')

    def test_full_drop_area_never_wraps_onto_occupied_slot(self):
        self.g.board.push_san('e4')
        self.g.pending = self.g.board.parse_san('d5')
        self.g.tasks = ['move the white pawn from e4 to the drop zone']
        self.g.drop_used = 6
        self.g.status = 'robot_wait'
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'paused')
        self.assertEqual(self.p.started, [])

    def test_stopped_promotion_can_resume_without_repeating_transfer(self):
        self.g.board = chess.Board('7k/8/8/8/8/8/p7/7K b - - 0 1')
        self.g.pending = self.g.board.parse_san('a1=Q')
        self.g.tasks = transfer_plan(self.g.board, self.g.pending)
        self.g.status = 'robot_running'
        self.staged_done()
        self.g.handle('live_confirm_transfer')
        self.g.handle('live_stop')
        self.g.handle('live_resume')
        self.assertEqual(self.g.status, 'promotion')
        self.assertEqual(self.p.started, [])
        self.g.handle('live_confirm_promotion')
        self.assertEqual(self.g.board.piece_at(chess.A1).symbol(), 'q')

    def test_delayed_camera_baseline_blocks_early_human_input(self):
        self.g.handle('live_calibrated')
        self.g.handle('live_start')
        self.assertEqual(self.g.status, 'watching')
        self.g.handle('live_human:e4')
        self.assertEqual(self.g.board.fen(), chess.STARTING_FEN)
        self.p.is_ready = False
        self.g.tick(time.monotonic()+2, np.zeros((8, 8, 3), np.uint8))
        self.assertEqual(self.g.status, 'watching')

    def test_real_stockfish_selects_legal_move(self):
        import shutil
        binary = shutil.which('stockfish')
        if not binary:
            self.skipTest('Stockfish is not installed')
        engine = Stockfish(binary, seconds=.02)
        self.addCleanup(engine.close)
        board = chess.Board()
        board.push_san('d4')
        self.assertIn(engine.choose(board), board.legal_moves)

    def test_engine_failure_is_recoverable_without_committing(self):
        def fail(board):
            raise RuntimeError('missing engine')
        self.g.engine.choose = fail
        self.start()
        self.g.handle('live_human:e4')
        with self.assertRaises(RuntimeError):
            self.g.future.result(timeout=2)
        self.g.tick(time.monotonic(), None)
        self.assertEqual(self.g.status, 'engine_error')
        self.assertEqual(self.g.board.peek().uci(), 'e2e4')
        self.assertEqual(self.p.started, [])


if __name__ == '__main__':
    unittest.main()

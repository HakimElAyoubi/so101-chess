"""Unscripted human-White/Stockfish-Black game using the staged learned player.

Physical transfers are transactions: the chess move is committed only after each
transfer has succeeded and the operator has checked the wrist view/physical board.
This supervisor does not claim independent occupancy or piece recognition.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine


def transfer_plan(board, move):
    """Expand a legal standard-chess move without committing its logical position."""
    if move not in board.legal_moves:
        raise ValueError('Move is not legal in the verified position.')
    piece = board.piece_at(move.from_square)
    colour = 'white' if piece.color else 'black'
    source, target = map(chess.square_name, (move.from_square, move.to_square))
    tasks = []
    if board.is_capture(move):
        square = move.to_square
        if board.is_en_passant(move):
            square += -8 if piece.color else 8
        captured = board.piece_at(square)
        tasks.append(f"move the {'white' if captured.color else 'black'} {chess.piece_name(captured.piece_type)} from {chess.square_name(square)} to the drop zone")
    tasks.append(f'move the {colour} {chess.piece_name(piece.piece_type)} from {source} to {target}')
    if board.is_castling(move):
        rank = source[1]
        rook_from, rook_to = ('h', 'f') if board.is_kingside_castling(move) else ('a', 'd')
        tasks.append(f'move the {colour} rook from {rook_from}{rank} to {rook_to}{rank}')
    return tasks


class Stockfish:
    """Owned by the thinking worker; engine I/O never blocks the arm control loop."""
    def __init__(self, path, seconds=.3):
        self.path, self.seconds, self.engine = path, seconds, None

    def choose(self, board):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path, timeout=10.)
            if 'Threads' in self.engine.options:
                self.engine.configure({'Threads': 1})
        move = self.engine.play(board, chess.engine.Limit(time=self.seconds)).move
        if move is None or move not in board.legal_moves:
            raise ValueError('Stockfish returned no legal move.')
        return move

    def close(self):
        if self.engine is not None:
            self.engine.quit()
            self.engine = None


class LiveGame:
    require_smolvla = True

    def __init__(self, player, engine=None, log_root=None):
        import so101_config
        self.p = player
        self.engine = engine or Stockfish(so101_config.STOCKFISH)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stockfish')
        self.future = None
        self.board = chess.Board()
        self.status = 'setup'
        self.message = 'Choose trained SmolVLA models, load them, calibrate, and set up the standard position.'
        self.pending = None
        self.tasks = []
        self.transfer_index = 0
        self.drop_used = 0
        self.log_root = Path(log_root or Path(__file__).parent / 'results/live_games')
        self.log_path = None
        self.history = []
        self.started = None
        self.evidence = {}
        self.observer = None
        self.baseline = None
        self.last_sample = 0.
        self.p_gain = 24
        self.calibrated = False

    def record(self, event, **data):
        row = dict(event=event, elapsed_s=round(time.monotonic()-self.started, 3) if self.started else 0.,
                   fen=self.board.fen(), transfer_index=self.transfer_index, **data)
        self.history.append(row)
        if self.log_path:
            with self.log_path.open('a') as f:
                f.write(json.dumps(row)+'\n')

    def profile(self):
        def details(policy, repo, checkpoint):
            return dict(repo=repo, checkpoint=checkpoint, type=getattr(policy, 'type', None),
                        resolved_path=str(getattr(policy, 'path', '')))
        return dict(placement=details(self.p.policy, self.p.model_repo, self.p.ckpt),
                    grasp=details(self.p.grasp_policy or self.p.policy, self.p.grasp_repo or self.p.model_repo,
                                  self.p.grasp_ckpt if self.p.grasp_repo else self.p.ckpt),
                    follower_p=self.p_gain, standin=bool(self.p.standin))

    def ready(self):
        return self.p.phase in ('idle', 'judging') and self.p.at_ready()

    def loaded(self):
        return (not self.p.loading and self.p.policy is not None and
                (self.p.standin or (getattr(self.p.policy, 'type', '') == 'smolvla' and
                 (not self.p.grasp_repo or getattr(self.p.grasp_policy, 'type', '') == 'smolvla'))))

    def handle(self, command):
        if command == 'stop' or command == 'live_stop':
            if self.status not in ('setup', 'complete', 'stopped'):
                self.stopped_from = self.status
                self.stopped_message = self.message
                self.status = 'stopped'
                self.message = 'Stopped. Reconcile any partial transfers before continuing or resetting.'
                self.record('stopped', pending=self.pending.uci() if self.pending else None)
            if command == 'live_stop':
                self.p.handle('stop')
                return True
            return False
        if command.startswith('live_models:'):
            if self.status not in ('setup', 'stopped', 'complete') or self.pending or self.p.loading or self.p.phase not in ('idle', 'judging'):
                self.message = 'Stop and reconcile the game before changing models.'
                return True
            try:
                data = json.loads(command.split(':', 1)[1])
                repo = str(data['repo']).strip()
                grasp = str(data.get('grasp_repo', '')).strip() or None
                ckpt = str(data.get('checkpoint', 'final')).strip()
                grasp_ckpt = str(data.get('grasp_checkpoint', ckpt)).strip()
                if not repo or any(':' in r for r in (repo, grasp) if r):
                    raise ValueError('Choose a SmolVLA Hub repository or local directory, not an API/scripted backend.')
                for value in (ckpt, grasp_ckpt):
                    if value != 'final' and not (value.isdigit() and len(value) == 6):
                        raise ValueError('Checkpoint must be final or a six-digit step, e.g. 007500.')
                self.p.model_repo, self.p.grasp_repo = repo, grasp
                self.p.grasp_ckpt = grasp_ckpt
                self.p.policy = self.p.grasp_policy = None
                self.p.standin = None
                self.p.load(ckpt)
                self.message = 'Loading selected models. Both skills must load successfully before starting.'
            except (ValueError, KeyError, TypeError) as exc:
                self.message = f'Could not select models: {exc}'
            return True
        if command == 'live_calibrated':
            if self.status == 'setup' and self.ready() and not getattr(self.p, '_measuring', False):
                self.calibrated = True
                self.message = 'Calibration acknowledged. Confirm all 32 starting pieces, then start.'
            return True
        if command == 'live_start':
            if self.status != 'setup':
                return True
            if not self.loaded() or not self.ready() or not self.calibrated or getattr(self.p, '_measuring', False):
                self.message = 'Load SmolVLA, return to ready, measure/check the board, then acknowledge calibration.'
                return True
            self.board = chess.Board()
            self.drop_used = self.p.drop_k = 0
            self.started = time.monotonic()
            directory = self.log_root / time.strftime('%Y%m%d_%H%M%S')
            directory.mkdir(parents=True, exist_ok=True)
            self.log_path = directory / f'events_{time.time_ns()}.jsonl'
            self.record('start', model=self.profile(), engine=getattr(self.engine, 'path', 'test engine'))
            self._human_turn()
            return True
        if command == 'live_reset':
            if self.status in ('setup', 'complete', 'stopped') and self.ready() and (not self.pending or self.board.turn == chess.WHITE):
                self.pending = None
                self.observer = None
                self.board = chess.Board()
                self.status = 'setup'
                self.calibrated = False
                self.log_path = None
                self.history = []
                self.message = 'Restore the starting position and clear all drop slots; check calibration again.'
            else:
                self.message = 'Resolve the pending move and return to ready before resetting.'
            return True
        if command.startswith('live_human:') or command.startswith('live_camera:'):
            if self.status != 'human_turn' or not self.ready():
                self.message = 'Wait for your turn with the arm settled at ready.'
                return True
            try:
                move = self.board.parse_san(command.split(':', 1)[1].strip())
                if self.board.turn != chess.WHITE:
                    raise ValueError('It is not White’s turn.')
                if command.startswith('live_camera:'):
                    if self.baseline is None:
                        raise ValueError('No settled baseline; wait at ready before moving.')
                    from human_move_watch import HumanMoveWatch
                    image, joints = self.baseline
                    self.observer = HumanMoveWatch(self.p.projector(), self.p._sess, self.board, move, image, joints)
                    self.pending = move
                    self.status = 'checking_human'
                    self.message = 'Checking source/destination changes. Remove your hand and hold still.'
                else:
                    self._accept_human(move, 'operator_confirmed')
            except ValueError as exc:
                self.message = f'Move not accepted: {exc}'
            return True
        if command == 'live_cancel_check' and self.status == 'checking_human':
            self.pending = None
            self.observer = None
            self.status = 'human_turn'
            self.message = 'Camera check cancelled. Enter the move and confirm it manually.'
            return True
        if command == 'live_think_retry' and self.status == 'engine_error':
            self.status = 'thinking'
            self.future = self.worker.submit(self.engine.choose, self.board.copy(stack=True))
            return True
        if command == 'live_clear_drop' and self.status in ('paused', 'stopped') and self.ready():
            self.drop_used = self.p.drop_k = 0
            self.record('drop_area_cleared_by_operator')
            self.message = 'Drop slots cleared. Restore the current transfer before retrying.'
            return True
        if command == 'live_resign' and self.status == 'human_turn' and self.ready():
            self.status = 'complete'
            self.message = 'White resigned. Black wins.'
            self.record('complete', result='0-1', termination='white_resignation')
            return True
        if command == 'live_claim_draw' and self.status == 'human_turn' and self.ready():
            if self.board.can_claim_draw():
                self.status = 'complete'
                self.message = 'Draw claim accepted.'
                self.record('complete', result='1/2-1/2', termination='claimed_draw')
            else:
                self.message = 'No legal draw claim in this position.'
            return True
        if command == 'live_confirm_transfer' and self.status == 'verify_transfer' and self.ready():
            self.record('transfer_verified_by_operator', task=self.tasks[self.transfer_index])
            if 'drop zone' in self.tasks[self.transfer_index]:
                self.drop_used += 1
            self.transfer_index += 1
            if self.transfer_index < len(self.tasks):
                self.status = 'robot_wait'
                self.message = 'Transfer verified; preparing the next part of the same move.'
            elif self.pending.promotion:
                self.status = 'promotion'
                self.message = f'Replace the pawn on {chess.square_name(self.pending.to_square)} with a black {chess.piece_name(self.pending.promotion)}; then confirm.'
            else:
                self._commit_robot()
            return True
        if command == 'live_confirm_promotion' and self.status == 'promotion' and self.ready():
            self.record('promotion_swap_by_operator', piece=chess.piece_name(self.pending.promotion))
            self._commit_robot()
            return True
        if command == 'live_resume' and self.status == 'stopped' and self.ready():
            previous = getattr(self, 'stopped_from', '')
            self.message = getattr(self, 'stopped_message', 'Resuming after inspection.')
            if previous == 'promotion':
                self.status = 'promotion'
            elif previous in ('human_turn', 'watching'):
                self._human_turn()
            elif previous == 'checking_human':
                self.pending, self.observer = None, None
                self.status = 'human_turn'
            elif previous in ('thinking', 'engine_error'):
                self.status = 'thinking'
                self.future = self.worker.submit(self.engine.choose, self.board.copy(stack=True))
            elif previous == 'verify_transfer':
                self.status = 'verify_transfer'
            elif previous in ('robot_running', 'robot_wait', 'paused'):
                self.status = 'paused'
                self.message = 'Inspect and restore the current transfer before retrying, or finish it by hand.'
            self.record('operator_resumed', previous=previous)
            return True
        if command == 'live_reconciled' and self.status in ('paused', 'stopped') and self.pending and not self.board.turn and self.ready() and self.transfer_index < len(self.tasks):
            self.status = 'verify_transfer'
            self.message = 'Check the current transfer’s destination, release and surrounding pieces, then confirm. This is recorded as assisted recovery.'
            self.record('manual_transfer_reconciliation', task=self.tasks[self.transfer_index])
            return True
        if command == 'live_retry' and self.status == 'paused' and self.pending and not self.board.turn and self.ready():
            # Operator explicitly restores the current transfer's source, not earlier completed transfers.
            self.status = 'robot_wait'
            self.message = 'Retry requested after restoring this transfer’s source and destination.'
            self.record('operator_retry', task=self.tasks[self.transfer_index])
            return True
        if command.startswith('live_'):
            return True
        # Allow calibration/manual controls only before a game, or while explicitly paused/stopped.
        if command == 'go_ready' or command == 'grid':
            return False
        if self.status in ('setup', 'complete', 'stopped', 'paused'):
            if command.startswith(('load:', 'custom:', 'select:', 'run', 'rec_')):
                self.message = 'Use the live-game controls for models and transfers.'
                return True
            if command.startswith(('calib', 'centred', 'point:', 'lower', 'raise', 'measure_board')):
                self.calibrated = False
            return False
        self.message = 'Stop the game before manual controls or calibration.'
        return True

    def _human_turn(self):
        if self._outcome():
            return
        self.status = 'watching'
        self.baseline = None
        self.baseline_at = time.monotonic() + 1.
        self.message = 'Wait for the arm and camera to settle at ready before moving.'

    def _accept_human(self, move, method):
        self.record('human_move_accepted', move=move.uci(), method=method, evidence=self.evidence)
        self.board.push(move)
        self.pending = None
        self.observer = None
        if self._outcome():
            return
        self.status = 'thinking'
        self.message = 'Stockfish is choosing Black’s reply.'
        self.future = self.worker.submit(self.engine.choose, self.board.copy(stack=True))

    def _outcome(self):
        outcome = self.board.outcome()
        if outcome:
            self.status = 'complete'
            self.message = f'Game over: {outcome.result()} ({outcome.termination.name.lower().replace("_", " ")}).'
            self.record('complete', result=outcome.result(), termination=outcome.termination.name)
            return True
        return False

    def tick(self, now, image):
        if self.status == 'watching':
            if not self.ready():
                self.baseline_at = now + 1.
            elif image is not None and now >= self.baseline_at:
                self.baseline = (image.copy(), self.p.q().copy())
                self.status = 'human_turn'
                self.message = 'Your turn: play any legal White move, enter it, then confirm. The arm stays at ready.'
        if self.status == 'checking_human' and self.ready() and image is not None and now-self.last_sample >= .25:
            self.last_sample = now
            self.evidence = self.observer.observe(image, self.p.q(), now)
            self.message = self.evidence['message']
            if self.evidence['confirmed']:
                self._accept_human(self.pending, 'camera_change_check')
        if self.status == 'thinking' and self.future and self.future.done():
            try:
                self.pending = self.future.result()
                self.tasks = transfer_plan(self.board, self.pending)
                self.transfer_index = 0
                self.record('engine_move_selected', move=self.pending.uci(), san=self.board.san(self.pending), tasks=self.tasks)
                self.status = 'robot_wait'
                self.message = f'Stockfish chose {self.board.san(self.pending)}.'
            except Exception as exc:
                self.pending = None
                self.status = 'engine_error'
                self.message = f'Stockfish failed: {exc}. Stop and reset, or retry thinking.'
                self.record('engine_error', reason=str(exc))
            self.future = None
        if self.status == 'robot_wait' and self.ready():
            task = self.tasks[self.transfer_index]
            if 'drop zone' in task:
                from policy_player import DROP_SPOTS
                if self.drop_used >= len(DROP_SPOTS):
                    self.status = 'paused'
                    self.message = 'All calibrated drop slots are occupied. Stop and clear the drop area; acknowledge it before retrying.'
                    return
                self.p.drop_k = self.drop_used
            self.p.replay_path = None
            self.p.phase = 'idle'
            self.p.last = None
            self.p.custom, self.p.trial = task, -1
            self.status = 'robot_running'
            self.record('transfer_started', move=self.pending.uci(), task=task, model=self.profile())
            self.p.start_trial()
            if self.p.phase != 'running':
                self.status = 'paused'
                self.message = f'Could not plan {task}. No chess move was committed.'
                self.record('transfer_planning_failed', task=task)
        if self.status == 'verify_transfer' and not self.ready() and self.p.phase in ('idle', 'judging'):
            self.p.handle('go_ready')

    def robot_finished(self, result):
        if self.status != 'robot_running':
            if self.status == 'stopped':
                self.record('interrupted_transfer_finished', result=result)
            return
        execution = result.get('staged_result')
        self.record('transfer_finished', result=result)
        self.p.phase = 'idle'
        if execution and execution[0] == 'placed':
            self.status = 'verify_transfer'
            self.message = 'Check that the piece was released upright at the destination and no neighbour moved; confirm to continue.'
        else:
            self.status = 'paused'
            self.message = 'Transfer failed. The chess move is still pending. Inspect the board; restore this transfer before retrying, or finish it manually and reconcile.'

    def _commit_robot(self):
        self.record('robot_move_verified', move=self.pending.uci())
        self.board.push(self.pending)
        self.pending = None
        self.tasks = []
        self.p.phase = 'idle'
        self.p.last = None
        self._human_turn()

    def state(self):
        return dict(mode='live', status=self.status, message=self.message, fen=self.board.fen(),
                    pieces={chess.square_name(s): p.symbol() for s, p in self.board.piece_map().items()},
                    legal_moves=[dict(uci=m.uci(), san=self.board.san(m)) for m in self.board.legal_moves] if self.board.turn else [],
                    pending=self.pending.uci() if self.pending else None, tasks=self.tasks, transfer_index=self.transfer_index,
                    drop_used=self.drop_used, model=self.profile(), calibrated=self.calibrated,
                    evidence=self.evidence, log=str(self.log_path or ''),
                    camera_baseline=self.baseline is not None, turn='White (you)' if self.board.turn else 'Black (robot)')

    def close(self):
        # Queue shutdown after any in-flight engine operation, preserving single-thread ownership.
        self.worker.submit(self.engine.close)
        self.worker.shutdown(wait=True, cancel_futures=False)

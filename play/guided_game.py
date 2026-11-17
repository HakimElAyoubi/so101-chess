"""Legal, paced guided chess game coordinated with the existing staged robot player."""
import json
import os
import time
from pathlib import Path
import chess
from human_move_watch import HumanMoveWatch

HERE=Path(__file__).resolve().parent


GAMES=HERE/'games'
SELECTED=GAMES/'.selected'          # the script last chosen on the page (survives a restart); GAME_SCRIPT still wins
DEFAULT_SCRIPT='six_move_demo'


def available_scripts():
    """Stems of the game scripts in play/games (a JSON with a 'san' move list), sorted."""
    out=[]
    for p in sorted(GAMES.glob('*.json')):
        try:
            if isinstance(json.loads(p.read_text()).get('san'),list):out.append(p.stem)
        except Exception:
            continue
    return out


def script_path(path=None):
    if path:return Path(path)
    if os.environ.get('GAME_SCRIPT'):return Path(os.environ['GAME_SCRIPT'])
    try:
        name=SELECTED.read_text().strip()
        if name and (GAMES/f'{name}.json').exists():return GAMES/f'{name}.json'
    except OSError:
        pass
    return GAMES/f'{DEFAULT_SCRIPT}.json'


def load_game(path=None):
    """The script as a list of plies: uci, san, the moved piece, the captured piece (None for a plain move) and the position
    before (fen) and after (fen_after) the ply, which the recorder draws as the board to set up for each take. Ordinary
    moves and ordinary captures only: castling, promotion and en passant are refused (two pieces or a piece that is not
    on the destination square), and the game must end in checkmate by Black, the arm."""
    spec=json.loads(script_path(path).read_text())
    board=chess.Board();line=[]
    for san in spec['san']:
        move=board.parse_san(san)
        if board.is_castling(move) or board.is_en_passant(move) or move.promotion:
            raise ValueError(f'{san}: the guided game allows ordinary moves and ordinary captures only (no castling, promotion or en passant).')
        taken=board.piece_at(move.to_square)
        line.append(dict(uci=move.uci(),san=board.san(move),piece=chess.piece_name(board.piece_at(move.from_square).piece_type),
                         capture=chess.piece_name(taken.piece_type) if taken else None,fen=board.fen()))
        board.push(move);line[-1]['fen_after']=board.fen()
    if not (board.is_checkmate() and board.turn==chess.WHITE):raise ValueError('The guided full game must end in checkmate by Black.')
    return spec,line


def describe(item,colour):
    """'move the black knight from c6 to e5' / 'take the white pawn off e5 and move the black knight from c6 to e5'."""
    move=chess.Move.from_uci(item['uci']);other='white' if colour=='black' else 'black'
    src,dst=chess.square_name(move.from_square),chess.square_name(move.to_square)
    text=f"move the {colour} {item['piece']} from {src} to {dst}"
    return f"take the {other} {item['capture']} off {dst} and {text}" if item['capture'] else text


class GuidedGame:
    def __init__(self,player,observer_factory=HumanMoveWatch,log_root=None,trust_human=None,script=None):
        self.p=player;self.spec,self.line=load_game(script);self.observer_factory=observer_factory;self.script=script_path(script).stem
        # 'play': the robot plays each move (a demonstration recorded with the leader is replayed when one exists for the
        # ply, else the vision grasp). 'record': on the robot's turns the user demonstrates the move with the leader arm.
        self.mode='play'
        # Trusted mode (default): pressing Space / the button takes the displayed White move as played, no camera
        # check. GAME_TRUST_HUMAN=0 restores the ready-pose before/after camera check (its motion threshold sits
        # below the real camera's noise floor, 1.1-2.7 on a still board, so it rarely confirms).
        self.trust_human=(os.environ.get('GAME_TRUST_HUMAN','1')!='0') if trust_human is None else bool(trust_human)
        self.board=chess.Board();self.ply=0;self.status='setup';self.message='Set up all 32 pieces in the normal starting position.'
        self.started=None;self.finished=None;self.observer=None;self.evidence={};self.last_sample=-1.;self.arm_since=None
        self.log_path=None;self.history=[];self.failed_reason=None;self.note='';self.last_failed_ply=None
        self.log_root=Path(log_root or HERE/'results/games')

    def record(self,event,**data):
        row=dict(event=event,elapsed_s=round(time.monotonic()-self.started,2) if self.started is not None else 0.,ply=self.ply,**data)
        self.history.append(row)
        if self.log_path:
            with self.log_path.open('a') as f:f.write(json.dumps(row)+'\n')

    def handle(self,command):
        now=time.monotonic()
        if command=='game_start':
            if self.status!='setup':return True
            if self.p.policy is None or self.p.grasp_policy is None:
                self.message='Wait for the grasp and placement policies to finish loading.';return True
            if self.p.phase!='idle' or not self.p.at_ready():
                self.message='Move the arm to ready (H), measure the board, then start.';return True
            self.board=chess.Board();self.ply=0;self.started=now
            directory=self.log_root/time.strftime('%Y%m%d_%H%M%S');directory.mkdir(parents=True,exist_ok=True)
            self.log_path=directory/'events.jsonl'
            (directory/'script.json').write_text(json.dumps(self.spec,indent=2)+'\n')
            self.status='human_baseline';self.arm_since=now;self.message='Hold still at ready while I take the starting view.'
            self.record('start');return True
        if command=='game_reset':
            if self.p.phase in ('idle','judging'):
                self.board=chess.Board();self.ply=0;self.status='setup';self.started=self.finished=None
                self.observer=None;self.evidence={};self.history=[];self.log_path=None
                self.p.phase='idle';self.message='Restore all 32 starting pieces before starting again.'
            return True
        if command=='game_check':
            if self.status=='human_turn':
                if self.trust_human:self._human_moved()
                else:self.status='checking_human';self.message='Checking your move. Remove your hand from the board.'
            return True
        if command=='game_retry':
            # Retry the robot move that just failed (available until the human plays on): the position is wound back
            # one ply and the move is planned again from its source square.
            if self.retry_available() and self.p.at_ready() and self.p.phase in ('idle','judging'):
                self.board.pop();self.ply-=1;self.note='';self.failed_reason=None;self.last_failed_ply=None
                self.p.phase='idle';self.status='robot_wait';self.message='Retrying the robot move from its source square.'
                self.record('retry_robot')
            elif self.retry_available():
                self.message='Bring the arm to ready (H) first, then retry.'
            return True
        if command=='game_stop' or (command=='stop' and self.status not in ('setup','complete','stopped')):
            self.status='stopped';self.message='Game stopped. The arm holds its current pose.';self.record('stopped')
            # Return False for the original stop so Player performs its normal hold.
            if command=='game_stop':self.p.handle('stop');return True
            return False
        if command.startswith('game_mode:'):
            if self.status in ('setup','complete','stopped'):self.mode='record' if command.endswith('record') else 'play'
            return True
        if command.startswith('game_script:'):
            # Choose another script (between games only): the board goes back to the setup position for it.
            name=command.split(':',1)[1]
            if self.status not in ('setup','complete','stopped') or self.p.phase not in ('idle','judging'):
                self.message='Finish or stop the game before choosing another script.';return True
            try:self.set_script(name)
            except Exception as exc:self.message=f'Could not load the script {name}: {exc}'
            return True
        if command.startswith('game_'):return True
        if command.startswith('rec_'):return False          # the player handles demonstration recording / play-back
        if self.status not in ('setup','complete','stopped','paused') and command not in ('grid','stop','go_ready'):
            self.message='Stop the guided game before calibration or other manual commands.';return True
        return False

    def tick(self,now,image):
        if self.status in ('setup','stopped','complete','paused'):return
        if self.status in ('human_baseline','human_turn','checking_human','robot_wait','robot_return'):
            if self.p.phase in ('homing','pointing'):return
            if not self.p.at_ready():
                self.message='Waiting for the arm to settle at ready; robot motion is blocked.'
                return
        if self.status=='robot_return':
            self.status='human_baseline';self.arm_since=now
        if self.status=='human_baseline':
            if image is None or now-self.arm_since<1.:return
            if self.trust_human:
                self.observer=None
            else:
                try:
                    move=chess.Move.from_uci(self.line[self.ply]['uci'])
                    self.observer=self.observer_factory(self.p.projector(),self.p._sess,self.board,move,image,self.p.q())
                except Exception as exc:
                    self.status='paused';self.message=f'Could not take the ready view: {exc}';self.record('camera_baseline_failed',reason=str(exc));return
            self.status='human_turn';self.evidence={}
            item=self.line[self.ply];move=chess.Move.from_uci(item['uci'])
            capture=(f"Capture: take the black {item['capture']} off {chess.square_name(move.to_square)} first, then put your {item['piece']} there. " if item['capture'] else '')
            self.message=(self.note+' ' if self.note else '')+capture+('Make the displayed White move, then press Space (or the button) when it is on its square.' if self.trust_human
                          else 'Make the displayed White move, remove your hand, then press Check my move.')
            self.record('human_turn_ready',expected=self.line[self.ply]['uci'])
        if self.status=='checking_human' and image is not None and now-self.last_sample>=.25:
            self.last_sample=now;self.evidence=self.observer.observe(image,self.p.q(),now)
            self.message=self.evidence['message']
            if self.evidence['confirmed']:self._human_moved(self.evidence)
        if self.status=='robot_wait':
            # No pacing: the robot starts as soon as the human move is in and the arm is at ready (target_seconds is only
            # the clock's label; the old 50 s-per-round wait made the player sit through a countdown).
            item=self.line[self.ply];move=chess.Move.from_uci(item['uci'])
            if move not in self.board.legal_moves:
                self.status='paused';self.message='Script no longer matches the chess position.';return
            if self.mode=='record':
                self.status='robot_record';self.note='';self.failed_reason=None
                what=(f"take the white {item['capture']} off {chess.square_name(move.to_square)} (drop it beside the board), then " if item['capture'] else '')
                self.message=(f"Record {item['san']}: hold the leader near the ready pose, press Record, {what}move the black {item['piece']} "
                              f"from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)} and bring the arm back up, then Stop & save.")
                self.record('robot_record_ready',move=item['uci']);return
            rec=self.recordings().get(self.ply)
            self.p.replay_path=Path(rec['path']) if rec else None
            if item['capture'] and not rec:
                # The vision grasp moves one piece onto an empty square; a capture (the white piece off first) is only
                # played from a demonstration. Counted as played, to be made by hand or retried after recording it.
                self.status='robot_running';self.record('robot_move_started',move=item['uci'])
                self._robot_done(None,'a capture needs a recorded demonstration');return
            self.p.trial=-1;self.p.custom=describe(item,'black')
            self.status='robot_running';self.note='';self.failed_reason=None
            self.message=f"Robot playing {item['san']}{' (replaying your demonstration)' if self.p.replay_path else ''}. Keep hands clear."
            self.record('robot_move_started',move=item['uci'])
            self.p.start_trial()
            if self.p.phase!='running':
                self._robot_done(None,'could not be planned')

    def _human_moved(self,evidence=None):
        """The displayed White move is on the board: seen by the camera (evidence) or taken on the player's word."""
        move=chess.Move.from_uci(self.line[self.ply]['uci']);self.board.push(move)
        if evidence:self.record('human_move_seen',move=move.uci(),evidence=evidence)
        else:self.record('human_move_assumed',move=move.uci())
        self.ply+=1;self.status='robot_wait';self.observer=None;self.message='Move taken as played.';self.last_failed_ply=None

    def robot_finished(self,result):
        if self.status!='robot_running':return
        completed=result.get('staged_result')
        if completed and completed[0]=='placed':self._robot_done(completed)
        else:self._robot_done(completed,result.get('end','unknown failure'))

    def _robot_done(self,execution,reason=None):
        """The robot's turn is over, placed or not: the game always goes on (only Stop ends it). After a failure the
        script's move is still counted as played and the next prompt asks for the piece to be put right by hand."""
        item=self.line[self.ply];move=chess.Move.from_uci(item['uci']);self.board.push(move)
        if reason is None:self.record('robot_move_completed',move=move.uci(),execution=execution)
        else:
            self.failed_reason=reason;self.record('robot_move_failed',move=move.uci(),reason=reason,execution=execution);self.last_failed_ply=self.ply
            by_hand=(f"take the white {item['capture']} off {chess.square_name(move.to_square)} and put the black {item['piece']} there by hand" if item['capture']
                     else f"put it on {chess.square_name(move.to_square)} by hand")
            self.note=(f"The robot could not finish {item['san']} ({reason}): press Retry robot move (put the black {item['piece']} back on "
                       f"{chess.square_name(move.from_square)} first if it moved), or {by_hand} and play on.")
        self.ply+=1;self.p.phase='idle';self.p.last=None
        if self.ply==len(self.line):
            self.status='complete';self.finished=time.monotonic()
            self.message=('Checkmate: Black, the arm, wins. The guided game is complete; the arm is back at ready.' if reason is None
                          else self.note+' That move is checkmate: Black, the arm, wins. The game is complete.')
            self.record('complete',checkmate=self.board.is_checkmate());return
        self.status='robot_return';self.arm_since=time.monotonic()
        if not self.p.at_ready():self.p.handle('go_ready')

    def set_script(self,name):
        """Load play/games/<name>.json and go back to the setup position; the choice is remembered for the next start."""
        path=GAMES/f'{name}.json'
        if not path.exists():raise FileNotFoundError(f'no play/games/{name}.json')
        self.spec,self.line=load_game(path);self.script=path.stem
        self.board=chess.Board();self.ply=0;self.status='setup';self.started=self.finished=None;self.observer=None;self.evidence={}
        self.history=[];self.log_path=None;self.note='';self.failed_reason=None;self.last_failed_ply=None
        self.message=f"{self.spec['name']} selected ({len(self.line)//2} moves each). Restore all 32 starting pieces, then start."
        try:SELECTED.write_text(name+'\n')
        except OSError:pass

    def recordings(self):
        from move_recorder import list_recordings
        return list_recordings(self.script)

    def recorded(self,ply,path,seconds):
        """The player saved a demonstration for ply: in record mode that is the robot's move."""
        if self.status=='robot_record' and ply==self.ply:
            self._robot_done(['recorded',f'{seconds:.1f} s',str(path)])

    def retry_available(self):
        return self.status=='human_turn' and self.last_failed_ply is not None and self.ply==self.last_failed_ply+1

    def failed_move(self):
        if not self.retry_available():return None
        item=self.line[self.last_failed_ply];move=chess.Move.from_uci(item['uci'])
        return dict(san=item['san'],piece=item['piece'],capture=item['capture'],source=chess.square_name(move.from_square),destination=chess.square_name(move.to_square))

    def state(self):
        elapsed=0. if self.started is None else (self.finished or time.monotonic())-self.started
        item=self.line[self.ply] if self.ply<len(self.line) else None
        move=chess.Move.from_uci(item['uci']) if item else None
        return dict(name=self.spec['name'],description=self.spec.get('description',''),status=self.status,message=self.message,elapsed_s=round(elapsed,1),
                    target_s=self.spec['target_seconds'],ply=self.ply,total_plies=len(self.line),rounds=len(self.line)//2,note=self.note,
                    round=min(len(self.line)//2,self.ply//2+1),turn='White (you)' if self.board.turn else 'Black (robot)',
                    move=item,source=chess.square_name(move.from_square) if move else None,
                    destination=chess.square_name(move.to_square) if move else None,evidence=self.evidence,
                    fen=self.board.fen(),pieces={chess.square_name(s):p.symbol() for s,p in self.board.piece_map().items()},
                    history=[h for h in self.history if h['event'] in ('human_move_seen','human_move_assumed','robot_move_completed')],failed_reason=self.failed_reason,
                    retry_available=self.retry_available(),failed_move=self.failed_move(),mode=self.mode,script=self.script,scripts=available_scripts(),
                    captures=sum(1 for it in self.line if it['capture']),
                    recordings={str(k):v for k,v in self.recordings().items()},
                    recording=(dict(ply=self.p.rec.ply,uci=self.p.rec.uci,seconds=round(self.p.rec.seconds,1)) if getattr(self.p,'rec',None) else None),
                    leader=bool(getattr(getattr(self.p,'leader',None),'connected',False)),
                    # for the recorder: the position to set up before the take, the one after it, and White's move just before
                    robot_moves=[dict(ply=i,san=it['san'],uci=it['uci'],piece=it['piece'],capture=it['capture'],fen=it['fen'],fen_after=it['fen_after'],
                                      white={k:self.line[i-1][k] for k in ('san','uci','piece','capture')}) for i,it in enumerate(self.line) if i%2==1])

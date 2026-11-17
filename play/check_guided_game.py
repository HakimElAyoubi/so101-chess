"""Offline guided-game legality, camera evidence and turn-coordination checks."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import chess
import cv2
import numpy as np
GAMES=Path(__file__).resolve().parent/'games'
os.environ['GAME_SCRIPT']=str(GAMES/'six_move_demo.json')      # the checks run on the six-move script whatever was selected on the page
import guided_game
from guided_game import GuidedGame,load_game
from human_move_watch import HumanMoveWatch

# From's gambit trap: ordinary captures by both sides, no castling, Black mates on move 6.
FROM_TRAP=['f4','e5','fxe5','d6','exd6','Bxd6','Nc3','Qh4+','g3','Qxg3+','hxg3','Bxg3#']


def write_script(directory,name,san,**extra):
    path=Path(directory)/f'{name}.json'
    path.write_text(json.dumps(dict(name=name,target_seconds=300,human_colour='white',description='',san=san,**extra)))
    return path


class ProjectorStub:
    def square_xyz(self,sq,session):
        f=ord(sq[0])-ord('a');r=int(sq[1])
        return np.array([(f+.5)*.1,(r-.5)*.1,.01])
    def project(self,q,points,width):
        points=np.asarray(points)
        return np.c_[points[:,0]*1000,(.8-points[:,1])*1000-points[:,2]*30],np.ones(len(points),bool)


def frame(*squares):
    image=np.full((800,800,3),80,np.uint8);P=ProjectorStub()
    for sq in squares:
        p=P.square_xyz(sq,None);uv,_=P.project(np.zeros(6),[p],800)
        cv2.circle(image,tuple(uv[0].astype(int)),6,(220,220,220),-1)
    return image


class HumanCameraChecks(unittest.TestCase):
    def watch(self):
        return HumanMoveWatch(ProjectorStub(),None,chess.Board(),chess.Move.from_uci('a2a3'),frame('a2'),np.zeros(6))
    def sample(self,watch,image):
        for t in np.arange(0,2,.25):result=watch.observe(image,np.zeros(6),float(t))
        return result
    def test_actual_source_to_destination_change_is_required(self):
        w=self.watch();result=self.sample(w,frame('a3'))
        self.assertTrue(result['confirmed']);self.assertEqual(result['source'],'a2');self.assertEqual(result['destination'],'a3')
    def test_wrong_destination_does_not_advance_even_though_script_knows_move(self):
        self.assertFalse(self.sample(self.watch(),frame('b3'))['confirmed'])
    def test_unchanged_board_or_exposure_shift_never_confirms(self):
        for image in (frame('a2'),np.clip(frame('a2').astype(int)+12,0,255).astype(np.uint8)):
            self.assertFalse(self.sample(self.watch(),image)['confirmed'])
    def test_moving_scene_and_arm_pose_change_block_confirmation(self):
        w=self.watch()
        for i in range(8):
            image=frame('a3') if i%2 else frame('a2')
            self.assertFalse(w.observe(image,np.zeros(6),i*.25)['confirmed'])
        shifted=np.zeros(6);shifted[0]=2.
        self.assertFalse(w.observe(frame('a3'),shifted,3.)['confirmed'])


class ObserverStub:
    def __init__(self,*args):self.confirm=False
    def observe(self,*args):return dict(confirmed=self.confirm,message='observed' if self.confirm else 'not observed')


class PlayerStub:
    def __init__(self):
        self.policy=self.grasp_policy=object();self.phase='idle';self.ready=True;self._sess='test'
        self.tasks=[];self.commands=[];self.last=None;self.trial=0;self.custom=''
    def at_ready(self):return self.ready
    def q(self):return np.zeros(6)
    def projector(self):return ProjectorStub()
    def start_trial(self):self.tasks.append(self.custom);self.phase='running'
    def handle(self,c):
        self.commands.append(c)
        if c=='go_ready':self.ready=True;self.phase='idle'
        elif c=='stop':self.phase='idle'


class GuidedGameChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.p=PlayerStub();self.clock=[0.]
        self.clock_patch=patch('guided_game.time.monotonic',side_effect=lambda:self.clock[0]);self.clock_patch.start();self.addCleanup(self.clock_patch.stop)
        self.game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=False)   # the camera-check path
    def tick(self,t):self.clock[0]=t;self.game.tick(t,np.zeros((8,8,3),np.uint8))
    def start(self):self.game.handle('game_start');self.tick(1.1)
    def test_full_game_is_legal_non_capturing_and_ends_in_mate(self):
        spec,line=load_game();self.assertEqual(len(line),12);self.assertEqual(spec['target_seconds'],300);self.assertFalse(any(i['capture'] for i in line))
        self.assertEqual(sorted({i['piece'] for i in line[1::2]}),['bishop','knight','pawn','queen'])
        board=chess.Board()
        for item in line:
            move=chess.Move.from_uci(item['uci']);self.assertIn(move,board.legal_moves)
            self.assertFalse(board.is_castling(move) or board.is_en_passant(move) or board.is_capture(move) or move.promotion)
            board.push(move)
        self.assertTrue(board.is_checkmate())
    def test_calibrated_ready_view_covers_all_human_moves_and_robot_targets_are_reachable(self):
        import sys
        sys.path[:0]=[str(Path(__file__).resolve().parent.parent/'goal')]
        from targets import Projector,LATEST
        from policy_player import READY6
        from staged import Planner
        P=Projector();Pl=Planner(P);image=np.zeros((720,1280,3),np.uint8)
        for script in ('six_move_demo','fifteen_move_game'):
            board=chess.Board()
            for item in load_game(GAMES/f'{script}.json')[1]:
                move=chess.Move.from_uci(item['uci'])
                with self.subTest(script=script,move=move.uci()):
                    if board.turn:
                        squares=(chess.square_name(move.from_square),chess.square_name(move.to_square))
                        if all(sq[1]!='1' for sq in squares):      # the ready view ends above d1-h1: first-rank White moves are taken on trust
                            watch=HumanMoveWatch(P,LATEST,board,move,image,READY6)
                            self.assertIn(watch.source,watch.masks);self.assertIn(watch.dest,watch.masks)
                    else:
                        source=P.square_xyz(chess.square_name(move.from_square),LATEST)
                        dest=P.square_xyz(chess.square_name(move.to_square),LATEST)
                        hover,_=Pl.grasp_hover(source,item['piece'],READY6,10.)
                        self.assertIsNotNone(hover)
                        if item['capture']:                                   # the white piece comes off the destination first
                            self.assertIsNotNone(Pl.grasp_hover(dest,item['capture'],READY6,10.)[0])
                        close_h=.0135 if item['piece']=='pawn' else .0171
                        place,_=Pl.place_hover(dest,close_h,hover,.8)
                        self.assertIsNotNone(place)
                board.push(move)

    def test_fifteen_move_game_is_legal_has_captures_on_both_sides_and_ends_in_black_mate(self):
        spec,line=load_game(GAMES/'fifteen_move_game.json');self.assertEqual(len(line),30);self.assertEqual(spec['target_seconds'],900)
        board=chess.Board()
        for item in line:
            move=chess.Move.from_uci(item['uci']);self.assertIn(move,board.legal_moves)
            self.assertFalse(board.is_castling(move) or board.is_en_passant(move) or move.promotion)
            self.assertEqual(bool(item['capture']),board.is_capture(move));self.assertEqual('x' in item['san'],bool(item['capture']))
            board.push(move)
        self.assertTrue(board.is_checkmate());self.assertEqual(board.result(),'0-1')
        self.assertEqual([i['capture'] for i in line[1::2] if i['capture']],['knight','queen','bishop','knight','pawn'])   # the arm's captures
        self.assertEqual([i['capture'] for i in line[0::2] if i['capture']],['knight','pawn'])                              # yours
        self.assertEqual(sorted({i['piece'] for i in line[1::2]}),['bishop','knight','pawn','queen'])
        game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=True,script=GAMES/'fifteen_move_game.json')
        st=game.state();self.assertEqual(st['rounds'],15);self.assertEqual(st['captures'],7);self.assertEqual(len(st['robot_moves']),15)
        self.assertIn('fifteen_move_game',st['scripts']);self.assertIn('six_move_demo',st['scripts'])
    def test_state_carries_the_position_before_and_after_each_robot_move_for_the_recorder_board(self):
        for name in ('six_move_demo','fifteen_move_game'):
            game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=True,script=GAMES/f'{name}.json');board=chess.Board()
            for r in game.state()['robot_moves']:
                while len(board.move_stack)<r['ply']:board.push_san(game.line[len(board.move_stack)]['san'])
                self.assertEqual(r['fen'],board.fen());self.assertEqual(r['white'],{k:game.line[r['ply']-1][k] for k in ('san','uci','piece','capture')})
                board.push_san(r['san']);self.assertEqual(r['fen_after'],board.fen())
        first=chess.Board(game.state()['robot_moves'][0]['fen'])          # 1...Nf6: White's knight already on f3, Black to move
        self.assertEqual(first.piece_at(chess.F3),chess.Piece(chess.KNIGHT,chess.WHITE));self.assertEqual(first.turn,chess.BLACK)
        nine=game.state()['robot_moves'][8];self.assertEqual((nine['san'],nine['capture'],nine['white']['san'],nine['white']['capture']),('fxe4','knight','Nxe4','knight'))
        self.assertIsNone(chess.Board(nine['fen_after']).piece_at(chess.F5));self.assertEqual(chess.Board(nine['fen_after']).piece_at(chess.E4),chess.Piece(chess.PAWN,chess.BLACK))

    def test_loader_accepts_ordinary_captures_and_refuses_castling_en_passant_and_white_mates(self):
        spec,line=load_game(write_script(self.temp.name,'from_trap',FROM_TRAP))
        self.assertEqual([i['capture'] for i in line],[None,None,'pawn',None,'pawn','pawn',None,None,None,'pawn','queen','pawn'])
        self.assertEqual(line[5]['piece'],'bishop');self.assertEqual(line[5]['san'],'Bxd6');self.assertEqual(line[5]['uci'],'f8d6')
        with self.assertRaisesRegex(ValueError,'castling'):load_game(write_script(self.temp.name,'castle',['e4','e5','Nf3','Nc6','Bc4','Bc5','O-O','Qh4']))
        with self.assertRaisesRegex(ValueError,'en passant'):load_game(write_script(self.temp.name,'ep',['e4','Nf6','e5','d5','exd6']))
        with self.assertRaisesRegex(ValueError,'by Black'):load_game(write_script(self.temp.name,'scholar',['e4','e5','Bc4','Nc6','Qh5','Nf6','Qxf7#']))
        with self.assertRaisesRegex(ValueError,'by Black'):load_game(write_script(self.temp.name,'short',['e4','e5']))
    def play_through(self,game,plies,robot=lambda ply:dict(staged_result=['placed','0 retries'])):
        """Trusted mode: alternate human Space and robot results until game.ply==plies (the clock advances 10 s a ply)."""
        while game.ply<plies:
            t=self.clock[0]+10.
            if game.status in ('human_baseline','robot_return'):self.tick(t);self.tick(t+1.1)
            if game.status=='human_turn':game.handle('game_check')
            elif game.status=='robot_wait':self.tick(t+2.)
            elif game.status=='robot_running':game.robot_finished(robot(game.ply))
            elif game.status=='robot_record':game.recorded(game.ply,Path(self.temp.name)/f'ply{game.ply:02d}.npz',12.)
            else:raise AssertionError(f'stuck in {game.status} at ply {game.ply}: {game.message}')
    def test_capture_prompts_the_human_and_a_robot_capture_without_a_take_is_left_to_the_hand(self):
        game=self.game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=True,script=write_script(self.temp.name,'from_trap',FROM_TRAP))
        self.assertEqual(game.script,'from_trap');game.handle('game_start');self.tick(1.1)
        self.play_through(game,2);self.tick(self.clock[0]+1.);self.tick(self.clock[0]+1.1);self.assertEqual(game.status,'human_turn')
        self.assertIn('take the black pawn off e5 first, then put your pawn there',game.message)       # 3. fxe5
        self.assertEqual(game.state()['move']['capture'],'pawn');self.assertEqual(self.p.tasks,['move the black pawn from e7 to e5'])
        self.play_through(game,5);self.assertEqual(game.status,'robot_wait');self.tick(self.clock[0]+2.)
        self.assertEqual(game.status,'robot_return');self.assertEqual(game.ply,6);self.assertEqual(game.failed_reason,'a capture needs a recorded demonstration')
        self.assertIn('take the white pawn off d6 and put the black bishop there by hand',game.note);self.assertEqual(len(self.p.tasks),2)   # no vision attempt
        self.assertEqual(game.history[-1]['event'],'robot_move_failed');self.assertTrue(game.state()['robot_moves'][2]['capture'])
        self.tick(self.clock[0]+1.);self.tick(self.clock[0]+1.1);self.assertTrue(game.retry_available());self.assertEqual(game.state()['failed_move']['capture'],'pawn')
        with patch.object(GuidedGame,'recordings',return_value={5:dict(uci='f8d6',path='/tmp/x/ply05_f8d6.npz',seconds=30.)}):
            game.handle('game_retry');self.assertEqual(game.ply,5);self.tick(self.clock[0]+2.)
            self.assertEqual(game.status,'robot_running');self.assertEqual(str(self.p.replay_path),'/tmp/x/ply05_f8d6.npz')
            self.assertEqual(self.p.tasks[-1],'take the white pawn off d6 and move the black bishop from f8 to d6')
        game.robot_finished(dict(staged_result=['placed','replayed 30.0 s']));self.assertEqual(game.ply,6);self.assertIsNone(game.failed_reason)
    def test_record_mode_capture_prompt_says_to_take_the_white_piece_off_first(self):
        game=self.game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=True,script=write_script(self.temp.name,'from_trap',FROM_TRAP))
        game.handle('game_mode:record');game.handle('game_start');self.tick(1.1);self.play_through(game,5);self.tick(self.clock[0]+2.)
        self.assertEqual(game.status,'robot_record')
        self.assertIn('take the white pawn off d6 (drop it beside the board), then move the black bishop from f8 to d6',game.message)
        self.play_through(game,12);self.assertEqual(game.status,'complete');self.assertTrue(game.board.is_checkmate())
        self.assertEqual(game.state()['captures'],6)
    def test_script_selector_switches_games_between_games_only_and_is_remembered(self):
        games=Path(self.temp.name)/'games';games.mkdir();write_script(games,'from_trap',FROM_TRAP);write_script(games,'six',json.loads((GAMES/'six_move_demo.json').read_text())['san'])
        (games/'notes.json').write_text('{"x":1}')
        with patch.object(guided_game,'GAMES',games),patch.object(guided_game,'SELECTED',games/'.selected'):
            self.assertEqual(guided_game.available_scripts(),['from_trap','six'])
            self.assertTrue(self.game.handle('game_script:from_trap'));self.assertEqual(self.game.script,'from_trap');self.assertEqual(len(self.game.line),12)
            self.assertEqual(self.game.status,'setup');self.assertIn('selected (6 moves each)',self.game.message);self.assertEqual((games/'.selected').read_text().strip(),'from_trap')
            self.assertEqual(self.game.state()['scripts'],['from_trap','six']);self.assertEqual(self.game.state()['captures'],6)
            self.game.handle('game_script:nothing');self.assertIn('Could not load the script nothing',self.game.message);self.assertEqual(self.game.script,'from_trap')
            self.start();self.game.handle('game_script:six');self.assertEqual(self.game.script,'from_trap');self.assertIn('Finish or stop the game',self.game.message)
            self.game.handle('game_stop');self.p.phase='running';self.game.handle('game_script:six');self.assertEqual(self.game.script,'from_trap')
            self.p.phase='idle';self.game.handle('game_script:six');self.assertEqual(self.game.script,'six');self.assertEqual(self.game.status,'setup')
            with patch.dict(os.environ,{},clear=False):
                os.environ.pop('GAME_SCRIPT',None);self.assertEqual(guided_game.script_path(),games/'six.json')       # remembered for the next start
                (games/'.selected').write_text('gone');self.assertEqual(guided_game.script_path().name,'six_move_demo.json')
    def test_camera_confirmation_gates_robot_motion_and_it_starts_at_once(self):
        self.start();self.assertEqual(self.game.status,'human_turn')
        self.game.handle('game_check');self.tick(2.)
        self.assertEqual(self.game.ply,0);self.assertEqual(self.p.tasks,[])
        self.game.observer.confirm=True;self.tick(3.)
        self.assertEqual(self.game.ply,1);self.assertEqual(self.game.status,'robot_running')   # no pacing wait
        self.assertEqual(self.p.tasks,['move the black pawn from e7 to e5'])
    def test_failed_robot_move_is_counted_as_played_and_the_game_goes_on(self):
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.tick(20.)
        self.game.robot_finished(dict(staged_result=['failed','empty'],end='empty grasp'))
        self.assertEqual(self.game.status,'robot_return');self.assertEqual(self.game.ply,2);self.assertEqual(self.game.failed_reason,'empty grasp')
        self.assertIn('Retry robot move',self.game.note);self.assertIn('e5 by hand',self.game.note);self.assertEqual(self.game.history[-1]['event'],'robot_move_failed')
        self.tick(21.1);self.tick(22.2);self.assertEqual(self.game.status,'human_turn');self.assertIn('by hand',self.game.message)
        self.assertTrue(self.game.state()['retry_available']);self.assertEqual(self.game.state()['failed_move']['source'],'e7')
        self.tick(23.);self.game.handle('game_check');self.game.observer.confirm=True;self.tick(70.)
        self.assertEqual(self.game.status,'robot_running');self.assertEqual(self.game.note,'');self.assertIsNone(self.game.failed_reason)
        self.assertFalse(self.game.retry_available())
    def test_retry_button_replays_the_failed_robot_move_until_the_human_plays_on(self):
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.tick(20.)
        fen_before=self.game.board.fen();self.game.robot_finished(dict(staged_result=['failed','empty'],end='empty grasp'))
        self.tick(21.1);self.tick(22.2);self.assertEqual(self.game.status,'human_turn');self.assertEqual(self.game.ply,2)
        self.p.ready=False;self.game.handle('game_retry');self.assertEqual(self.game.ply,2);self.assertIn('ready',self.game.message)
        self.p.ready=True;self.game.handle('game_retry');self.assertEqual(self.game.ply,1);self.assertEqual(self.game.board.fen(),fen_before)
        self.assertEqual(self.game.status,'robot_wait');self.assertIsNone(self.game.failed_reason);self.assertFalse(self.game.retry_available())
        self.tick(23.);self.assertEqual(self.game.status,'robot_running');self.assertEqual(self.p.tasks,['move the black pawn from e7 to e5']*2)
    def test_unplannable_robot_move_is_counted_as_played_and_the_game_goes_on(self):
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.p.start_trial=lambda:None;self.tick(20.)
        self.assertEqual(self.game.status,'robot_return');self.assertEqual(self.game.ply,2);self.assertEqual(self.game.failed_reason,'could not be planned')
    def test_all_plies_follow_camera_robot_return_cycle_to_checkmate(self):
        self.start();rounds=len(self.game.line)//2
        for turn in range(rounds):
            if turn:
                self.tick(turn*50+1.1)
                self.tick(turn*50+2.2)
            self.assertEqual(self.game.status,'human_turn')
            self.game.handle('game_check');self.game.observer.confirm=True
            self.tick(turn*50+3.)
            self.tick(turn*50+20.)
            self.assertEqual(self.game.status,'robot_running')
            self.clock[0]=turn*50+49.
            self.game.robot_finished(dict(staged_result=['placed','0 retries']))
        self.assertEqual(len(self.p.tasks),rounds);self.assertEqual(self.game.status,'complete')
        self.assertTrue(self.game.board.is_checkmate());self.assertLessEqual(self.game.state()['elapsed_s'],self.game.spec['target_seconds'])
    def test_trusted_mode_takes_the_move_as_played_without_camera_evidence(self):
        self.game=GuidedGame(self.p,ObserverStub,Path(self.temp.name),trust_human=True)
        self.start();self.assertEqual(self.game.status,'human_turn');self.assertIsNone(self.game.observer)
        self.game.handle('game_check');self.assertEqual(self.game.ply,1);self.assertEqual(self.game.status,'robot_wait')
        self.assertEqual(self.game.history[-1]['event'],'human_move_assumed');self.assertEqual(len(self.game.state()['history']),1)
        self.tick(20.);self.assertEqual(self.game.status,'robot_running');self.assertEqual(self.p.tasks,['move the black pawn from e7 to e5'])
    def test_default_mode_is_trusted_unless_env_disables_it(self):
        import os
        with patch.dict(os.environ,{},clear=False):
            os.environ.pop('GAME_TRUST_HUMAN',None);self.assertTrue(GuidedGame(self.p,ObserverStub,Path(self.temp.name)).trust_human)
            os.environ['GAME_TRUST_HUMAN']='0';self.assertFalse(GuidedGame(self.p,ObserverStub,Path(self.temp.name)).trust_human)
    def test_unready_arm_blocks_human_observation_and_robot_start(self):
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.p.ready=False
        self.tick(25.);self.assertEqual(self.game.ply,0);self.assertEqual(self.p.tasks,[])
    def test_successful_robot_return_rehomes_before_human_view(self):
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.tick(20.)
        self.p.ready=False;self.game.robot_finished(dict(staged_result=['placed','0 retries']))
        self.assertIn('go_ready',self.p.commands)
        self.tick(21.);self.assertEqual(self.game.status,'human_baseline')
        self.tick(22.1);self.assertEqual(self.game.status,'human_turn')
    def test_record_mode_waits_for_a_demonstration_and_counts_it_as_the_robot_move(self):
        self.game.handle('game_mode:record');self.assertEqual(self.game.mode,'record')
        self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.tick(20.)
        self.assertEqual(self.game.status,'robot_record');self.assertEqual(self.p.tasks,[])
        self.assertIn('e7',self.game.message);self.assertTrue(self.game.handle('game_mode:play'));self.assertEqual(self.game.mode,'record')   # not in setup: ignored
        self.assertFalse(self.game.handle('rec_start:1:e7e5'))        # rec_ commands go through to the player
        self.game.recorded(1,Path(self.temp.name)/'ply01_e7e5.npz',9.5)
        self.assertEqual(self.game.ply,2);self.assertEqual(self.game.status,'robot_return');self.assertEqual(self.game.history[-1]['execution'][0],'recorded')
        st=self.game.state();self.assertEqual(st['mode'],'record');self.assertEqual(len(st['robot_moves']),6);self.assertIsNone(st['recording'])
    def test_play_mode_replays_a_recorded_demonstration_when_one_exists(self):
        with patch.object(GuidedGame,'recordings',return_value={1:dict(uci='e7e5',path='/tmp/x/ply01_e7e5.npz',seconds=9.5)}):
            self.start();self.game.handle('game_check');self.game.observer.confirm=True;self.tick(20.)
            self.assertEqual(self.game.status,'robot_running');self.assertEqual(str(self.p.replay_path),'/tmp/x/ply01_e7e5.npz')
            self.assertIn('replaying your demonstration',self.game.message)
        self.game.robot_finished(dict(staged_result=['placed','replayed 9.5 s']))
        with patch.object(GuidedGame,'recordings',return_value={}):      # no recording for ply 3: the vision grasp
            self.assertEqual(self.game.ply,2);self.tick(21.1);self.tick(22.2);self.game.handle('game_check');self.game.observer.confirm=True;self.tick(70.)
            self.assertIsNone(self.p.replay_path)
    def test_replay_move_eases_in_plays_the_recording_and_returns_home(self):
        from move_recorder import ReplayMove,MoveRecording
        ready=np.array([0.,-60.,40.,90.,20.,2.]);rec=MoveRecording(Path(self.temp.name)/'ply01_e7e5.npz',1,'e7e5',0.)
        for i in range(0,61):
            t=i/30.;q=ready+np.array([5.,20.,-10.,-15.,0.,25.])*np.sin(np.pi*t/2.)**2
            rec.add(t,q,q)
        path=rec.save();mv=ReplayMove(path,ready,0.)
        q=ready.copy();t=0.;out=[]
        while mv.result is None and t<30.:
            a=mv.step(q,None,t)
            if a is not None:q=np.asarray(a,float);out.append((t,mv.stage,q.copy()))
            t+=1/30.
        self.assertEqual(mv.result[0],'placed');self.assertIn('replayed 2.0 s',mv.result[1])
        self.assertEqual([e[1] for e in mv.events],['to_start','replay','return','done'])
        stages=[o[1] for o in out];self.assertLess(stages.index('replay'),stages.index('return'))
        mid=[o[2] for o in out if o[1]=='replay'];self.assertGreater(max(m[5] for m in mid),20.)   # the gripper opened as recorded
        np.testing.assert_allclose(out[-1][2],ready,atol=.5)             # back at ready
        self.assertTrue(mv.on_path())
    def test_stop_holds_and_manual_commands_are_blocked_while_active(self):
        self.start();self.assertTrue(self.game.handle('nudge:left'))
        self.game.handle('game_stop');self.assertEqual(self.game.status,'stopped');self.assertIn('stop',self.p.commands)


if __name__=='__main__':unittest.main(verbosity=2)

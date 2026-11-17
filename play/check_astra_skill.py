"""Offline API-contract and full simulated-move checks. Never opens hardware or calls an API."""
import copy
import io
import json
import sys
import unittest
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / 'goal')]
import astra_skill as A
import llm_skill as _L
_L.MID_LOOK_VERIFIED=False   # the recorded bishop/queen fixtures below encode the straight verified descent; the real arm uses the mid-descent look
import llm_skill as L
from policy_io import load_policy

ANSWER = dict(visible=True, target_px=[420, 400], pad_gap_px=[580, 350],
              clear=True, roll_deg=0, confidence=0.9, between_pads=False)
CROP = np.zeros((840, 840, 3), np.uint8)


def response(answer=None, **changes):
    r = dict(status='completed', model='gpt-6-astra', output=[dict(type='reasoning'),
             dict(type='message', content=[dict(type='output_text', text=json.dumps(ANSWER if answer is None else answer))])],
             usage=dict(input_tokens=100, output_tokens=50, input_tokens_details=dict(cached_tokens=20)))
    r.update(changes)
    return r


def bare_skill():
    sk = A.AstraSkill.__new__(A.AstraSkill)
    sk.model, sk.effort, sk._key, sk.cost_calls = 'gpt-6-astra', 'medium', 'test-key', 0
    sk.service_tier = 'default'
    sk._cost_ledger=A.SpendLedger(move_limit=1.)
    return sk


class ApiChecks(unittest.TestCase):
    def test_payload_and_measurement(self):
        sk = bare_skill()
        with patch.object(sk, '_post', return_value=response()) as post:
            ans = sk._model_answer('GRASP target', CROP)
        payload = post.call_args.args[0]
        self.assertEqual(payload['model'], 'gpt-6-astra')
        self.assertFalse(payload['store'])
        self.assertTrue(payload['text']['format']['strict'])
        self.assertEqual(payload['instructions'], A.SYSTEM_PROMPT)
        self.assertTrue(payload['input'][0]['content'][0]['image_url'].startswith('data:image/png;base64,iVBOR'))
        self.assertEqual(ans['target_px'], ANSWER['target_px'])
        self.assertEqual(ans['_usage']['cached'], 20)

    def test_truncation_is_charged_and_not_automatically_resubmitted(self):
        sk=bare_skill()
        truncated=dict(status='incomplete',incomplete_details=dict(reason='max_output_tokens'),
                       output=[],usage=dict(input_tokens=100,output_tokens=1024))
        with patch.object(sk,'_post',return_value=truncated) as post:
            with self.assertRaises(RuntimeError):sk._model_answer('look',CROP)
        self.assertEqual(post.call_count,1)
        self.assertEqual(post.call_args.args[0]['max_output_tokens'],1024)
        self.assertGreater(sk._cost_ledger.spent,0.)

    def test_refusal_incomplete_and_bad_json_rejected(self):
        for r in [response(output=[dict(type='message', content=[dict(type='refusal', refusal='no')])]),
                  response(status='incomplete', incomplete_details=dict(reason='content_filter')),
                  response(output=[dict(type='message', content=[dict(type='output_text', text='not JSON')])])]:
            with self.subTest(response=r), patch.object(A.AstraSkill, '_post', return_value=r):
                with self.assertRaises((ValueError, RuntimeError)):
                    bare_skill()._model_answer('look', CROP)

    def test_invalid_pixels_and_flags_rejected(self):
        for field, value in [('target_px', [float('nan'), 100]), ('target_px', [-1, 100]),
                             ('target_px', [840, 100]), ('target_px', None), ('pad_gap_px', [0, float('inf')]),
                             ('pad_gap_px', [0]), ('roll_deg', 91), ('confidence', 2), ('visible', 'true'), ('between_pads','true')]:
            ans = copy.deepcopy(ANSWER); ans[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                A.validate_answer(ans, CROP)
        ans = copy.deepcopy(ANSWER); ans.update(visible=False, target_px=None)
        A.validate_answer(ans, CROP)

    def test_https_transport(self):
        sk = bare_skill()
        ctx = io.BytesIO(json.dumps(response()).encode())
        with patch.object(A, 'urlopen', return_value=ctx) as urlopen:
            self.assertEqual(sk._post({'model': sk.model})['status'], 'completed')
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, 'https://api.openai.com/v1/responses')
        self.assertEqual(req.get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(urlopen.call_args.kwargs['timeout'], 35.0)

    def test_transport_failures_do_not_leak_secrets(self):
        for error in [A.HTTPError('https://api.openai.com/v1/responses', 401, 'test-key', {}, None),
                      A.URLError('test-key'), TimeoutError('test-key')]:
            with patch.object(A, 'urlopen', side_effect=error), self.assertRaises(RuntimeError) as caught:
                bare_skill()._post({})
            self.assertNotIn('test-key', str(caught.exception))

    def test_claude_transport_still_decodes(self):
        sk = L.ClaudeSkill.__new__(L.ClaudeSkill)
        resp = SimpleNamespace(content=[SimpleNamespace(type='text', text=json.dumps(ANSWER))],
                               stop_reason='end_turn', usage=SimpleNamespace(input_tokens=10, output_tokens=20))
        create = unittest.mock.Mock(return_value=resp)
        sk.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        sk.model, sk.cost_calls = L.MODEL_DEFAULT, 0
        self.assertEqual(sk._model_answer('look', CROP)['target_px'], ANSWER['target_px'])
        self.assertEqual(create.call_args.kwargs['model'], L.MODEL_DEFAULT)


class TrackingChecks(unittest.TestCase):
    def test_tracks_translations_without_a_model_call(self):
        from astra_tracking import FeatureTracker
        rng = np.random.default_rng(42)
        gray = np.full((240, 320), 100, np.uint8)
        gray[75:125, 125:175] = rng.integers(0, 255, (50, 50), dtype=np.uint8)
        tracker = FeatureTracker(gray, [150, 100])
        self.assertTrue(tracker.valid)
        for i in range(1, 10):
            moved = cv2.warpAffine(gray, np.float32([[1, 0, 2*i], [0, 1, -i]]), (320, 240), borderValue=100)
            self.assertTrue(tracker.update(moved), tracker.reason)
        np.testing.assert_allclose(tracker.pixel, [168, 91], atol=.3)
        self.assertFalse(tracker.update(np.full_like(gray, 100)))

    def test_weak_texture_and_expired_tracks_rejected(self):
        from astra_tracking import FeatureTracker
        self.assertFalse(FeatureTracker(np.zeros((100, 100), np.uint8), [50, 50]).valid)
        gray = np.random.default_rng(1).integers(0, 255, (100, 100), dtype=np.uint8)
        tracker = FeatureTracker(gray, [50, 50])
        tracker.started -= 11
        self.assertFalse(tracker.update(gray))

    def test_local_corrections_skip_api_and_new_height_reacquires(self):
        sk = A.AstraSkill(repo='astra:mock'); sk.mock = False
        gray = np.full((720, 1280), 100, np.uint8)
        gray[75:125, 125:175] = np.random.default_rng(5).integers(0,255,(50,50),dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        meta = dict(x0=0,y0=0,scale=1.5,pads_px=[580,350])
        sk.annotate = lambda *args: (CROP, meta)
        sk._situation_text = lambda *args: 'GRASP'
        sk._grid_pixels = lambda *args: []
        observed = dict(ANSWER, target_px=[225,150])
        with patch.object(sk, '_model_answer', return_value=observed) as model:
            initial,_,_,_ = sk._ask(np.zeros(6),image,'from')
            moved = cv2.warpAffine(image,np.float32([[1,0,2],[0,1,-1]]),(1280,720),borderValue=(100,100,100))
            local,_,_,_ = sk._ask(np.zeros(6),moved,'from')
            self.assertEqual(initial['_source'],'astra')
            self.assertEqual(local['_source'],'local_tracker')
            np.testing.assert_allclose(local['target_px'],[228,148.5],atol=.4)
            self.assertEqual(model.call_count,1)
            sk.low = True
            bottom,_,_,_ = sk._ask(np.zeros(6),moved,'from')
            self.assertEqual(bottom['_source'],'astra')
            self.assertEqual(model.call_count,2)


class ControlChecks(unittest.TestCase):
    def test_worker_from_previous_attempt_cannot_replace_current_answer(self):
        entered, release = threading.Event(), threading.Event()
        threads = []
        RealThread = threading.Thread
        def make_thread(*args, **kwargs):
            t = RealThread(*args, **kwargs); threads.append(t); return t
        sk = L.ClaudeSkill.__new__(L.ClaudeSkill); sk.reset()
        calls = []
        def ask(*args):
            n = len(calls) + 1; calls.append(n)
            if n == 1:
                entered.set(); release.wait(2)
            return {"call": n}, {}, CROP, "look"
        sk._ask = ask
        with patch.object(L.threading, 'Thread', side_effect=make_thread):
            sk._start_look(np.zeros(6), CROP, 'from')
            self.assertTrue(entered.wait(1))
            sk.reset()
            sk._start_look(np.zeros(6), CROP, 'from')
            threads[1].join(1)
            release.set(); threads[0].join(1)
        self.assertTrue(sk.pending['done'])
        self.assertEqual(sk.pending['ans']['call'], 2)

    def test_astra_will_not_close_when_look_budget_is_exhausted(self):
        sk = A.AstraSkill(repo='astra:mock')
        sk.goal = {'which': ('from',)}
        sk.phase, sk.low, sk.look_n = 'settle_look_low', True, sk.max_looks
        sk.desired, sk.bias, sk.grip = np.zeros(6), np.zeros(6), 15.0
        sk.settle_start, sk.settle_t0 = -2, -2
        action = sk.act(np.zeros(6), CROP, 'pick up')
        self.assertEqual(sk.phase, 'give_up')
        self.assertIsNotNone(sk.gave_up)
        self.assertEqual(action[5], 15.0)

    def test_blocked_uncertain_or_unconverged_bottom_cannot_close(self):
        for clear, confidence, distance in [(False,.95,.5),(True,.4,.5),(True,.95,5)]:
            with self.subTest(clear=clear,confidence=confidence,distance=distance):
                sk = A.AstraSkill(repo='astra:mock')
                sk.goal = {'which':('from',)}
                sk.phase,sk.low,sk.moves_low = 'look',True,2
                sk.desired,sk.bias,sk.grip = np.zeros(6),np.zeros(6),15.
                ans = dict(visible=True,target_px=[100,100],clear=clear,confidence=confidence)
                meta = dict(q=[0]*6,land_px=[100,100],land_w=[0,0,0],z_plane=.03,tips_px=[])
                sk.pending = dict(done=True,ans=ans,meta=meta,q=np.zeros(6),crop=CROP,text='look')
                with patch.object(sk,'truth_xy',return_value=None), patch.object(sk,'register',return_value={}), \
                     patch.object(sk,'measure',return_value=(distance,0.,{})), patch.object(sk,'_log_look'):
                    action = sk.act(np.zeros(6),CROP,'pick up')
                self.assertIn(sk.phase,('give_up','settle_look_low'))   # 2026-11-10: a low-confidence look is re-taken (never closed on)
                self.assertNotEqual(sk.phase,'closed');self.assertEqual(action[5],15.)

    def test_recorded_pawn_body_alignment_sends_close_instead_of_more_nudges(self):
        rows = json.loads((HERE / 'fixtures/astra_pawn_body_20261110.json').read_text())
        for row in rows:
            with self.subTest(look=row['n']):
                sk = A.AstraSkill(repo='astra:mock')
                sk.low,sk.phase,sk.moves_low = True,'look',0
                q = np.array(row['meta']['q'])
                sk.desired,sk.bias,sk.grip = q.copy(),np.zeros(6),15.
                sk.goal = {'which':('from',)}
                sk.staged = SimpleNamespace(piece='pawn',from_xyz=np.array([0,0,row['meta']['z_board']]),
                                            to_xyz=np.array([0,0,.01]))
                sk.pending = dict(done=True,ans=row['answer'],meta=row['meta'],q=q,crop=CROP,text='recorded look')
                with patch.object(sk,'truth_xy',return_value=None),patch.object(sk,'register',return_value={}), \
                     patch.object(sk,'measure',return_value=(*row['d_mm'],{})),patch.object(sk,'_log_look') as log:
                    action = sk.act(q,CROP,'pick up pawn')
                self.assertEqual(sk.phase,'closed')
                self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
                self.assertTrue(log.call_args.args[1]['capture_window']['ok'])
                self.assertEqual(sk.moves_low,0)

    def test_capture_window_rejects_cross_jaw_error_and_wrong_height(self):
        row = json.loads((HERE / 'fixtures/astra_pawn_body_20261110.json').read_text())[0]
        sk = A.AstraSkill(repo='astra:mock');sk.low=True
        sk.staged = SimpleNamespace(piece='pawn',from_xyz=np.array([0,0,.01]),to_xyz=np.array([0,0,.01]))
        q = np.array(row['meta']['q']);_,a,b=sk.tips(q)
        axis=(b-a)[:2];axis/=np.linalg.norm(axis);across=np.array([-axis[1],axis[0]])
        self.assertTrue(sk.grasp_capture(row['answer'],row['meta'],*(axis*3))['ok'])
        self.assertFalse(sk.grasp_capture(row['answer'],row['meta'],*(across*3))['ok'])
        self.assertFalse(sk.grasp_capture(row['answer'],row['meta'],*(axis*5))['ok'])
        high = dict(row['meta'],z_board=row['meta']['z_board']-.01)
        self.assertFalse(sk.grasp_capture(row['answer'],high,*(axis*3))['ok'])
        for key,value in [('visible',False),('clear',False),('confidence',.5)]:
            ans=dict(row['answer']);ans[key]=value
            self.assertFalse(sk.grasp_capture(ans,row['meta'],*(axis*3))['ok'])

    def test_anchor_without_grid_residual_does_not_break_place(self):
        sk = A.AstraSkill(repo='astra:mock')
        sk.goal = {'which': ('to',)}
        sk._reg().update(looks=1, n=0, resid_mm=None)
        meta = dict(q=[0]*6, z_plane=.01, land_w=[.1,.2,.01], square='e5',
                    centre_w=[.1,.2,.01], scale=1.5, x0=0, y0=0)
        with patch.object(sk, 'backproject', return_value=np.array([.101,.202])):
            dx, dy, info = sk.measure({'target_px':[100,100]}, meta, 'to')
        self.assertAlmostEqual(dx, 1)
        self.assertAlmostEqual(dy, 2)
        self.assertEqual(info['method'], 'centre')


class KnightChecks(unittest.TestCase):
    def test_quarter_turn_at_both_knight_squares_before_any_visual_alignment(self):
        from policy_player import READY6
        for square in ('b8','g8'):
            with self.subTest(square=square):
                sk = A.AstraSkill(repo='astra:mock')
                xyz = sk.proj.square_xyz(square,sk.session)
                q,_ = sk.Pl.grasp_hover(xyz,'knight',READY6,10.)
                sk.staged = SimpleNamespace(piece='knight',colour='black',task=f'move the black knight from {square} to f6',
                                            from_xyz=xyz,to_xyz=sk.proj.square_xyz('f6',sk.session))
                sk.phase,sk.raised,sk.grip = 'settle_look',True,10.
                sk.desired,sk.bias = q.copy(),np.zeros(6)
                sk.settle_start,sk.settle_t0 = -10.,-10.
                with patch.object(A.time,'monotonic',return_value=0.):
                    self.assertIsNotNone(sk._orient_knight(q))
                    raised = sk.move_to.copy()
                    self.assertTrue(sk._within_joint_limits(raised))
                    self.assertFalse(sk._knight_turned)
                    sk.desired = raised.copy();sk.move_t0=None
                    sk.settle_start,sk.settle_t0=-10.,-10.
                    self.assertIsNotNone(sk._orient_knight(raised))
                    turned=sk.move_to.copy()
                    self.assertAlmostEqual(abs(turned[4]-raised[4]),90.,places=6)
                    self.assertTrue(sk._within_joint_limits(turned))
                    self.assertLess(1000*np.linalg.norm(sk.tips(turned)[0]-sk.tips(raised)[0]),2.)
                    self.assertGreaterEqual(sk.tips(raised)[0][2]-xyz[2]-.031,.017)
                    self.assertGreaterEqual(sk.move_T,5.6)
                    self.assertTrue(sk._knight_turned)
                    self.assertIsNone(sk.pending)
                    self.assertIsNone(sk._orient_knight(turned))
                text=sk._situation_text('from',{})
                self.assertIn('90-degree quarter-turn is complete',text)
                self.assertIn('roll_deg=0',text)
                sk.reset()
                self.assertFalse(sk._knight_turned)

    def test_model_cannot_undo_the_completed_knight_turn(self):
        sk=A.AstraSkill(repo='astra:mock');sk.mock=False;sk._knight_turned=True
        sk.annotate=lambda *args:(CROP,dict(x0=0,y0=0,scale=1.5,pads_px=[580,350]))
        sk._situation_text=lambda *args:'KNIGHT quarter-turn complete'
        sk._grid_pixels=lambda *args:[]
        sk._targets=lambda:(np.zeros(3),np.zeros(3),'knight')
        observed=dict(ANSWER,roll_deg=-45)
        with patch.object(sk,'_model_answer',return_value=observed):
            ans,_,_,_=sk._ask(np.zeros(6),np.zeros((720,1280,3),np.uint8),'from')
        self.assertEqual(ans['roll_deg'],0.)


class QueenChecks(unittest.TestCase):
    def queen_descent(self, fixture="astra_queen_hover_20261110.json", answer_changes=None, physical=False):
        row=json.loads((HERE/'fixtures'/fixture).read_text())
        sk=A.AstraSkill(repo='astra:mock');sk.mock=not physical
        sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.array([0,0,row['z_board']]),
                                  to_xyz=np.array([0,0,row['z_board']]),task='move the black queen from d8 to a5')
        sk.goal={'which':('from',),'move':sk.staged.task}
        sk.desired=np.array(row['q']);sk.bias=np.zeros(6);sk.grip=10.
        sk.z_board=row['z_board'];sk.answer=dict(row['answer'], **(answer_changes or {}));sk.last_dist=float(np.linalg.norm(row['d_mm']))
        sk._down('from',0.)
        return sk,row

    def test_physical_queen_trim_is_not_chased_in_high_hover_measurements(self):
        sk=A.AstraSkill(repo='astra:mock');sk.mock=False
        sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.zeros(3),
                                  to_xyz=np.zeros(3),task='move the black queen from d8 to a5')
        with patch.object(L.ClaudeSkill,'measure',side_effect=lambda *args: (0.,0.,{})):
            dx,dy,info=sk.measure({}, {}, 'from')
        np.testing.assert_allclose([dx,dy],[0.,0.])
        self.assertNotIn('queen_file_trim_mm',info)

    def test_failed_trim_trial_would_converge_without_chasing_offset_at_hover(self):
        row=json.loads((HERE/'fixtures/astra_queen_trim_hover_20261110.json').read_text())
        sk=A.AstraSkill(repo='astra:mock');sk.mock=False
        sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.zeros(3),to_xyz=np.zeros(3))
        r=row['record'];ans=r['answer'];q=np.array(r['q'])
        px=np.array(r['target_px']);raw=np.array(r['target_raw'])
        origin=raw-px/1.5
        meta=dict(q=r['q'],land_w=r['land_w'],z_plane=r['z_plane'],scale=1.5,
                  x0=origin[0],y0=origin[1],tips_mid=sk.tips(q)[0].tolist(),
                  pads_px=(np.array(ans['pad_gap_px'])-ans['pads_offset_px']).tolist())
        dx,dy,_=sk.measure(ans,meta,'from')
        self.assertGreater(np.linalg.norm(r['d_mm']),2.)
        self.assertLess(np.hypot(dx,dy),2.)
        np.testing.assert_allclose([dx,dy],np.array(r['d_mm'])-r['queen_trim_world_mm'],atol=1e-6)

    def test_physical_queen_descends_on_original_hover_axis_without_extra_step(self):
        sk,row=self.queen_descent(physical=True)
        self.assertIsNone(sk.gave_up)
        cert=sk._verified_descent
        start=sk.tips(np.array(row['q']))[0]
        np.testing.assert_allclose(cert['xy'],start[:2],atol=1e-9)
        self.assertNotIn('remaining',cert)
        q=sk.move_to.copy();sk.desired=q.copy();sk.move_t0=None
        sk.phase='settle_look_low';sk.settle_start=sk.settle_t0=-10.
        with patch.object(A.time,'monotonic',return_value=cert['deadline']-1), \
             patch.object(sk,'_start_look',side_effect=AssertionError('unexpected bottom look')), \
             patch.object(sk,'annotate',return_value=(CROP,{})),patch.object(sk,'_log_look'),patch.object(sk,'anchor'):
            action=sk.act(q,CROP,'pick up queen')
        self.assertEqual(sk.phase,'closed')
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
        self.assertLess(1000*np.linalg.norm(sk.tips(q)[0][:2]-start[:2]),.8)

    def test_recorded_unreachable_command_rebases_descent_on_measured_pose(self):
        row=json.loads((HERE/'fixtures/astra_queen_unreachable_command_20261110.json').read_text())
        sk=A.AstraSkill(repo='astra:mock')
        sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.array([0,0,row['z_board']]),
                                  to_xyz=np.array([0,0,row['z_board']]),task='move the black queen from d8 to g4')
        sk.goal={'which':('from',),'move':sk.staged.task}
        command=np.array(row['command']);actual=np.array(row['q'])
        sk.desired=command.copy();sk.bias=np.array([0.,0.,0.,4.,0.,0.]);sk.grip=10.
        sk.z_board=row['z_board'];sk.answer=row['answer'];sk.last_dist=float(np.linalg.norm(row['d_mm']))
        # Reproduce the former rejection from the unachieved command.
        sk._down('from',0.)
        if sk.gave_up:   # with LAND 10 mm from the fixed jaw (2026-11-10) this recorded command is reachable; the rebase below still runs
            self.assertIn('cannot preserve queen alignment',sk.gave_up)
        sk.gave_up=None;sk.desired=command.copy();sk.bias=np.array([0.,0.,0.,4.,0.,0.])
        sk._measured_q6=actual.copy()
        sk._down('from',0.)
        self.assertIsNone(sk.gave_up)
        self.assertTrue(sk._within_joint_limits(sk.move_to))
        np.testing.assert_allclose(sk.move_from,actual)
        np.testing.assert_array_equal(sk.bias,np.zeros(6))
        np.testing.assert_allclose(sk._verified_descent['xy'],sk.tips(actual)[0][:2])
        q=sk.move_to.copy();sk.desired=q.copy();sk.move_t0=None
        sk.phase='settle_look_low';sk.settle_start=sk.settle_t0=-10.
        with patch.object(A.time,'monotonic',return_value=sk._verified_descent['deadline']-1), \
             patch.object(sk,'annotate',return_value=(CROP,{})),patch.object(sk,'_log_look'),patch.object(sk,'anchor'):
            action=sk.act(q,CROP,'pick up queen')
        self.assertEqual(sk.phase,'closed')
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
        self.assertLess(1000*np.linalg.norm(sk.tips(q)[0][:2]-sk.tips(actual)[0][:2]),.8)

    def replay_hover(self, row, skill=None):
        sk=skill or A.AstraSkill(repo='astra:mock')
        q=np.array(row['q'])
        sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.array([0,0,.01]),
                                  to_xyz=np.array([0,0,.01]),task='move the black queen from d8 to g4')
        sk.goal={'which':('from',),'move':sk.staged.task}
        sk.desired=q.copy();sk.bias=np.zeros(6);sk.grip=10.;sk.z_board=.01
        sk.phase='look';sk.raised=True;sk.low=False
        sk.pending=dict(done=True,ans=row['answer'],meta=row,q=q,crop=CROP,text='recorded hover')
        with patch.object(sk,'truth_xy',return_value=None),patch.object(sk,'register',return_value={}), \
             patch.object(sk,'measure',return_value=(*row['d_mm'],{})),patch.object(sk,'_log_look') as log:
            action=sk.act(q,CROP,'pick up queen')
        return sk,action,log.call_args.args[1]

    def test_recorded_082_hover_rechecks_then_descends_on_fresh_confirmation(self):
        rows=json.loads((HERE/'fixtures/astra_queen_hover_retries_20261110.json').read_text())['records']
        sk,action,log=self.replay_hover(rows[2])
        self.assertIsNone(sk.gave_up)
        self.assertEqual(log['action'],'relook')
        self.assertEqual(sk.phase,'settle_look')
        self.assertFalse(sk.low)
        self.assertIsNone(sk.move_t0)
        self.assertAlmostEqual(float(action[5]),10.)
        row=copy.deepcopy(rows[2]);row['answer']['confidence']=.90
        sk,_,log=self.replay_hover(row,sk)
        self.assertEqual(log['action'],'descend')
        self.assertIsNotNone(sk._verified_descent)
        self.assertIsNone(sk.gave_up)

    def test_queen_confidence_rechecks_are_bounded_and_never_close(self):
        row=json.loads((HERE/'fixtures/astra_queen_hover_retries_20261110.json').read_text())['records'][2]
        sk=None
        for expected in ('relook','relook','give_up'):
            sk,action,log=self.replay_hover(row,sk)
            self.assertEqual(log['action'],expected)
            self.assertFalse(sk.low)
            self.assertAlmostEqual(float(action[5]),10.)
        self.assertIn('confidence not confirmed',sk.gave_up)

    def test_recorded_hover_exhaustion_gets_two_bounded_corrections(self):
        row=json.loads((HERE/'fixtures/astra_queen_hover_retries_20261110.json').read_text())['records'][7]
        sk=A.AstraSkill(repo='astra:mock');sk.moves_hover=4
        with patch.object(sk,'_lateral',return_value=tuple(row['d_mm'])):
            for expected in ('move','move','give_up'):
                sk,_,log=self.replay_hover(row,sk)
                self.assertEqual(log['action'],expected)
        self.assertFalse(sk.low)
        self.assertEqual(sk.moves_hover,6)

    def test_physical_queen_hover_corrections_from_recorded_poses_are_reachable(self):
        rows=json.loads((HERE/'fixtures/astra_queen_hover_retries_20261110.json').read_text())['records']
        for row in (rows[0],rows[1],rows[4],rows[5],rows[6],rows[7]):
            with self.subTest(look=row['n']):
                sk=A.AstraSkill(repo='astra:mock');sk.mock=False
                q=np.array(row['q']);sk.desired=q.copy();sk.bias=np.zeros(6);sk.grip=10.
                sk.staged=SimpleNamespace(piece='queen',colour='black',from_xyz=np.array([0,0,.01]),to_xyz=np.zeros(3))
                sk.goal={'which':('from',)};sk._measured_q6=q.copy()
                start=sk.tips(q)[0]
                dx,dy=sk._lateral(q,*row['d_mm'],0.)
                self.assertIsNone(sk.gave_up)
                self.assertTrue(sk._within_joint_limits(sk.move_to))
                target=start+np.array([dx,dy,0.])/1000
                self.assertLess(1000*np.linalg.norm(sk.tips(sk.move_to)[0][:2]-target[:2]),.6)
                self.assertGreaterEqual(sk.tips(sk.move_to)[0][2],.065-.001)
                self.assertLessEqual(start[2]-sk.tips(sk.move_to)[0][2],.0156)
                self.assertGreater(np.hypot(dx,dy),0.)
                # Check actual interpolation stays above the crown during hover.
                for u in np.linspace(0,1,11):
                    self.assertGreater(sk.tips(sk.move_from+(sk.move_to-sk.move_from)*u)[0][2],.060)
                np.testing.assert_allclose([dx,dy],row['d_mm'])
                # The recovered hover must also admit the subsequent body grasp.
                sk.desired=sk.move_to.copy();sk._measured_q6=sk.desired.copy()
                sk.move_t0=None;sk.answer=dict(ANSWER,confidence=.9);sk.last_dist=.5;sk.z_board=.01
                sk._down('from',10.)
                self.assertIsNone(sk.gave_up)
                self.assertIsNotNone(sk._verified_descent)
                self.assertTrue(sk._within_joint_limits(sk.move_to))

    def test_queen_recovery_preserves_blocked_and_look_budget_vetoes(self):
        rows=json.loads((HERE/'fixtures/astra_queen_hover_retries_20261110.json').read_text())['records']
        row=copy.deepcopy(rows[2]);row['answer']['clear']=False
        sk,_,log=self.replay_hover(row)
        self.assertEqual(log['action'],'give_up')
        self.assertFalse(sk.low)
        sk=A.AstraSkill(repo='astra:mock');sk.look_n=sk.max_looks-1
        sk,_,log=self.replay_hover(rows[2],sk)
        self.assertEqual(log['action'],'give_up')
        self.assertFalse(sk.low)

    def test_physical_queen_command_does_not_wind_up_against_wrist_limit(self):
        row=json.loads((HERE/'fixtures/astra_queen_unreachable_command_20261110.json').read_text())
        sk=A.AstraSkill(repo='astra:mock');sk.mock=False
        sk.staged=SimpleNamespace(piece='queen',colour='black');sk.goal={'which':('from',)}
        sk.desired=np.array(row['command']);sk.bias=np.zeros(6);sk.grip=10.
        q=np.array(row['q'])
        for _ in range(100):
            cmd=sk._command(q)
            model=sk.proj.C.lerobot_to_qpos(cmd)[:5]
            self.assertTrue(np.all(model<=sk.Pl.K.hi+1e-6))
            self.assertTrue(np.all(model>=sk.Pl.K.lo-1e-6))
        self.assertLess(abs(sk.bias[3]),5.)

    def test_only_black_queen_height_is_lowered_by_eleven_mm(self):
        sk=A.AstraSkill(repo='astra:mock');sk.staged=SimpleNamespace(colour='black')
        self.assertAlmostEqual(sk.grasp_height_cm('queen'),1.71)
        self.assertAlmostEqual(sk.grasp_height_cm('pawn'),1.35)
        self.assertAlmostEqual(sk.grasp_height_cm('knight'),1.71)
        sk.staged.colour='white'
        self.assertAlmostEqual(sk.grasp_height_cm('queen'),2.81)

    def test_recorded_aligned_hover_descends_lower_and_closes_without_another_api_look(self):
        sk,row=self.queen_descent()
        self.assertIsNotNone(sk._verified_descent)
        q=sk.move_to.copy();sk.desired=q.copy();sk.move_t0=None
        sk.phase='settle_look_low';sk.settle_start=sk.settle_t0=-10.
        site,_=sk.Pl.fk(q)
        self.assertLess(abs(1000*(site[2]-row['z_board'])-17.1),.8)
        self.assertLess(1000*np.linalg.norm(sk.tips(q)[0][:2]-sk._verified_descent['xy']),.8)
        now=sk._verified_descent['deadline']-1
        with patch.object(A.time,'monotonic',return_value=now),patch.object(sk,'_model_answer',side_effect=AssertionError('unnecessary API look')), \
             patch.object(sk,'annotate',return_value=(CROP,{})),patch.object(sk,'_log_look') as log,patch.object(sk,'anchor'):
            action=sk.act(q,CROP,'pick up queen')
        self.assertEqual(sk.phase,'closed')
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
        self.assertEqual(log.call_args.args[1]['answer']['_source'],'verified_descent')

    def test_recorded_088_hover_uses_verified_close_path(self):
        sk,row=self.queen_descent("astra_queen_hover_088_20261110.json")
        self.assertIsNotNone(sk._verified_descent)
        q=sk.move_to.copy();sk.desired=q.copy();sk.move_t0=None
        sk.phase='settle_look_low';sk.settle_start=sk.settle_t0=-10.
        self.assertLess(1000*np.linalg.norm(sk.tips(q)[0][:2]-sk._verified_descent['xy']),.8)
        with patch.object(A.time,'monotonic',return_value=sk._verified_descent['deadline']-1), \
             patch.object(sk,'_start_look',side_effect=AssertionError('bottom look would chase occluded crown')), \
             patch.object(sk,'annotate',return_value=(CROP,{})),patch.object(sk,'_log_look'),patch.object(sk,'anchor'):
            action=sk.act(q,CROP,'pick up queen')
        self.assertEqual(sk.phase,'closed')
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)

    def test_unverified_queen_hover_never_descends_into_bottom_corrections(self):
        for changes in ({'confidence':.84},{'clear':False},{'visible':False}):
            with self.subTest(changes=changes):
                sk,_=self.queen_descent(answer_changes=changes)
                self.assertIsNone(sk._verified_descent)
                self.assertIsNone(sk.move_t0)
                self.assertFalse(sk.low)
                self.assertIn('hover not verified',sk.gave_up)

    def test_queen_cannot_move_sideways_at_body_height(self):
        sk,_=self.queen_descent()
        sk.move_t0=None
        before=sk.desired.copy()
        self.assertEqual(sk._lateral(before,3.4,-6.5,0.),(0.,0.))
        np.testing.assert_array_equal(sk.desired,before)
        self.assertIsNone(sk.move_t0)
        self.assertIn('no lateral queen corrections',sk.gave_up)

    def test_wrong_height_does_not_authorize_queen_close(self):
        sk,row=self.queen_descent()
        q=np.array(row['q']) # still at the high hover: descent did not happen
        sk.desired=sk.move_to.copy();sk.move_t0=None;sk.phase='settle_look_low'
        sk.settle_start=sk.settle_t0=-10.
        with patch.object(A.time,'monotonic',return_value=sk._verified_descent['deadline']+.1):
            action=sk.act(q,CROP,'pick up queen')
        self.assertEqual(sk.phase,'give_up')
        self.assertIn('queen descent did not settle',sk.gave_up)
        self.assertGreater(float(action[5]),6.)

    def test_old_grasp_anchor_cannot_mark_current_queen_as_wrong_piece(self):
        sk=A.AstraSkill(repo='astra:mock')
        sk._reg().update(looks=1,n=0,resid_mm=None)
        centre=sk.square_centre_w('d8')
        meta=dict(q=[0]*6,z_plane=.047,land_w=centre.tolist(),square='d8',centre_w=centre.tolist(),
                  scale=1.5,x0=0,y0=0)
        feature=centre[:2]+[.017,0]
        ans=dict(target_px=[100,100],grid_px=[],_grid_trusted=False)
        with patch.object(sk,'backproject',return_value=feature):
            _,_,info=sk.measure(ans,meta,'from')
            self.assertFalse(info.get('wrong_piece',False))
            ans['_grid_trusted']=True
            _,_,info=sk.measure(ans,meta,'from')
            self.assertTrue(info['wrong_piece'])

    def test_sparse_or_collinear_corners_cannot_translate_board(self):
        sk=A.AstraSkill(repo='astra:mock');before=sk._reg()['t'].copy()
        for points in [[[10,10]],[[10,10],[100,10]],[[10,10],[50,10],[100,10]]]:
            ans={'grid_px':points}
            info=sk.register(ans,{})
            self.assertEqual(info['grid_used'],0)
            self.assertFalse(ans['_grid_trusted'])
            np.testing.assert_array_equal(sk._reg()['t'],before)


class BishopChecks(unittest.TestCase):
    def bishop_descent(self, attempt, colour='black', **changes):
        row=attempt['hover'];q=np.array(row['q'])
        sk=A.AstraSkill(repo='astra:mock');sk.mock=False
        sk.staged=SimpleNamespace(piece='bishop',colour=colour,from_xyz=np.array([0,0,.01]),
                                  to_xyz=np.array([0,0,.01]),task=f'move the {colour} bishop from c8 to h3')
        sk.goal={'which':('from',),'move':sk.staged.task}
        sk.desired=q.copy();sk.desired[3]+=4.;sk.bias=np.array([0.,0.,0.,4.,0.,0.])
        sk._measured_q6=q.copy();sk.grip=10.;sk.z_board=.01
        sk.answer=dict(row['answer'],**changes);sk.last_dist=float(np.linalg.norm(row['d_mm']))
        sk._down('from',0.)
        return sk,q

    def attempts(self):
        return json.loads((HERE/'fixtures/astra_bishop_occlusion_20261110.json').read_text())['attempts']

    def test_recorded_bishop_hovers_close_after_descent_without_occluded_top_look(self):
        for attempt in self.attempts():
            for colour in ('black','white'):
                with self.subTest(look=attempt['hover']['n'],colour=colour):
                    self.assertFalse(attempt['occluded']['answer']['visible'])
                    sk,hover=self.bishop_descent(attempt,colour)
                    self.assertIsNone(sk.gave_up)
                    cert=sk._verified_descent
                    self.assertEqual(cert['piece'],'bishop')
                    self.assertAlmostEqual(sk.grasp_height_cm('bishop'),2.27)
                    np.testing.assert_allclose(sk.move_from,hover)
                    self.assertTrue(sk._within_joint_limits(sk.move_to))
                    q=sk.move_to.copy();sk.desired=q.copy();sk.move_t0=None
                    sk.phase='settle_look_low';sk.settle_start=sk.settle_t0=-10.
                    site,_=sk.Pl.fk(q)
                    self.assertLess(abs((site[2]-.01)*1000-22.7),.8)
                    with patch.object(A.time,'monotonic',return_value=cert['deadline']-1), \
                         patch.object(sk,'_start_look',side_effect=AssertionError('occluded top would veto grasp')), \
                         patch.object(sk,'annotate',return_value=(CROP,{})),patch.object(sk,'_log_look') as log,patch.object(sk,'anchor'):
                        action=sk.act(q,CROP,'pick up bishop')
                    self.assertEqual(sk.phase,'closed')
                    self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
                    self.assertEqual(log.call_args.args[1]['piece'],'bishop')
                    self.assertLess(1000*np.linalg.norm(sk.tips(q)[0][:2]-sk.tips(hover)[0][:2]),.8)

    def test_unverified_bishop_never_descends(self):
        for change in ({'confidence':.84},{'visible':False},{'clear':False}):
            with self.subTest(change=change):
                sk,_=self.bishop_descent(self.attempts()[0],**change)
                self.assertIsNone(sk._verified_descent)
                self.assertIsNone(sk.move_t0)
                self.assertFalse(sk.low)
                self.assertIn('bishop hover not verified',sk.gave_up)

    def test_bishop_not_at_body_height_never_closes(self):
        sk,hover=self.bishop_descent(self.attempts()[0])
        sk.desired=sk.move_to.copy();sk.move_t0=None;sk.phase='settle_look_low'
        sk.settle_start=sk.settle_t0=-10.
        with patch.object(A.time,'monotonic',return_value=sk._verified_descent['deadline']+.1):
            action=sk.act(hover,CROP,'pick up bishop')
        self.assertEqual(sk.phase,'give_up')
        self.assertIn('bishop descent did not settle',sk.gave_up)
        self.assertGreater(float(action[5]),10.)

    def test_bishop_cannot_nudge_sideways_at_body_height(self):
        sk,_=self.bishop_descent(self.attempts()[0]);sk.move_t0=None
        before=sk.desired.copy()
        self.assertEqual(sk._lateral(before,3.,-4.,0.),(0.,0.))
        np.testing.assert_array_equal(sk.desired,before)
        self.assertIn('no lateral bishop corrections',sk.gave_up)


class VisualContactChecks(unittest.TestCase):
    def body_record(self):
        return json.loads((HERE/'fixtures/astra_bishop_occlusion_20261110.json').read_text())['attempts'][0]['occluded']

    def test_fresh_body_between_pads_closes_without_top_pixel_or_alignment_move(self):
        row=self.body_record();q=np.array(row['q'])
        for piece in ('bishop','queen','pawn','knight','rook','king'):
            with self.subTest(piece=piece):
                sk=A.AstraSkill(repo='astra:mock')
                sk.staged=SimpleNamespace(piece=piece,colour='black')
                sk.goal={'which':('from',)};sk.phase='look';sk.low=False;sk.raised=True
                sk.desired=q.copy();sk.bias=np.zeros(6);sk.grip=15.;sk.look_n=sk.max_looks-1
                ans=dict(ANSWER,visible=False,target_px=None,between_pads=True,confidence=.9,_source='astra')
                sk.pending=dict(done=True,ans=ans,meta=row,q=q,crop=CROP,text='body between pads')
                with patch.object(sk,'measure',side_effect=AssertionError('unnecessary pixel centering')), \
                     patch.object(sk,'_down',side_effect=AssertionError('unnecessary descent')), \
                     patch.object(sk,'_log_look') as log,patch.object(sk,'anchor'):
                    action=sk.act(q,CROP,'pick up piece')
                self.assertEqual(sk.phase,'closed')
                self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
                np.testing.assert_allclose(action[:5],q[:5],atol=1e-4)
                self.assertEqual(log.call_args.args[1]['method'],'fresh visual body between contact pads')

    def test_contact_signal_rejects_hover_projection_cached_or_uncertain_observation(self):
        row=self.body_record();q=np.array(row['q']);sk=A.AstraSkill(repo='astra:mock')
        valid=dict(ANSWER,between_pads=True,confidence=.9,_source='astra')
        for changes in ({'between_pads':False},{'_source':'local_tracker'}, {'confidence':.84},
                        {'clear':False},{'pad_gap_px':None}):
            with self.subTest(changes=changes):
                self.assertFalse(sk.visual_grasp_ready(dict(valid,**changes),row,q,'from'))
        self.assertFalse(sk.visual_grasp_ready(valid,row,q,'to'))
        moved=q.copy();moved[0]+=2.
        self.assertFalse(sk.visual_grasp_ready(valid,row,moved,'from'))

    def test_schema_accepts_occluded_top_with_visible_body_contact(self):
        ans=dict(ANSWER,visible=False,target_px=None,between_pads=True,confidence=.9)
        self.assertTrue(A.validate_answer(ans,CROP)['between_pads'])


class ApiRecoveryChecks(unittest.TestCase):
    def skill(self):
        sk=A.AstraSkill(repo='astra:mock');sk.goal={'which':('from',)}
        sk.staged=SimpleNamespace(piece='bishop',colour='black',from_xyz=np.zeros(3),to_xyz=np.zeros(3))
        sk.desired=np.zeros(6);sk.bias=np.zeros(6);sk.grip=15.;sk._measured_q6=np.zeros(6)
        return sk

    def test_429_holds_without_spending_visual_budget_and_reacquires_current_frame(self):
        sk=self.skill();sk.phase='look';sk.raised=True
        sk.pending=dict(done=True,error='OpenAI HTTP 429',api_status=429,api_retry_after=12.,api_fatal=False)
        frame=np.zeros((40,40,3),np.uint8)
        with patch.object(A.time,'monotonic',return_value=100.),patch.object(A.random,'uniform',return_value=0.),patch.object(sk,'_log_look'):
            action=sk.act(np.zeros(6),frame,'grasp')
        self.assertEqual(sk.phase,'api_backoff')
        self.assertEqual(sk.look_n,0)
        self.assertEqual(sk._api_wait_until,112.)
        self.assertAlmostEqual(float(action[5]),15.)
        with patch.object(A.time,'monotonic',return_value=111.),patch.object(sk,'_start_look') as start:
            sk.act(np.zeros(6),frame,'grasp');start.assert_not_called()
        current=np.full_like(frame,120)
        with patch.object(A.time,'monotonic',return_value=113.),patch.object(sk,'_settled',return_value=True),patch.object(sk,'_start_look') as start:
            sk.act(np.zeros(6),current,'grasp')
        self.assertIs(start.call_args.args[1],current)
        self.assertEqual(sk.phase,'look')

    def test_persistent_429_quota_or_long_retry_delay_stops_without_arm_retries(self):
        for case in ('persistent','quota','long_delay'):
            with self.subTest(case=case):
                sk=self.skill()
                p=dict(api_status=429,api_retry_after=120. if case=='long_delay' else None,api_fatal=case=='quota')
                with patch.object(A.random,'uniform',return_value=0.):
                    for now in (0.,5.,15.,35.):
                        sk.look_n=1;sk.handle_look_error(p,now,'from')
                        if sk.fatal_error:break
                self.assertIsNotNone(sk.fatal_error)
                self.assertIsNone(sk.gave_up)
                import staged as S
                for stage in ('grasp','place'):
                    mv=S.StagedMove.__new__(S.StagedMove)
                    mv.stage=stage;mv.t0=0.;mv.result=None;mv.events=[]
                    mv.skill=SimpleNamespace(step=lambda *args:np.array([0,0,0,0,0,15.]),fatal_error=sk.fatal_error)
                    with patch.object(mv,'_retry',side_effect=AssertionError('API outage must not move arm')):
                        self.assertIsNone(mv.step(np.zeros(6),CROP,40.))
                    self.assertEqual(mv.result,('failed',sk.fatal_error))

    def test_429_transport_preserves_retry_after_and_distinguishes_account_quota(self):
        for code,fatal in (('rate_limit_exceeded',False),('insufficient_quota',True)):
            body=io.BytesIO(json.dumps({'error':{'code':code,'message':'test-key'}}).encode())
            error=A.HTTPError('https://api.openai.com/v1/responses',429,'test-key',{'Retry-After':'17'},body)
            with patch.object(A,'urlopen',side_effect=error),self.assertRaises(A.AstraAPIError) as caught:
                bare_skill()._post({})
            self.assertEqual(caught.exception.retry_after_s,17.)
            self.assertEqual(caught.exception.fatal,fatal)
            self.assertNotIn('test-key',str(caught.exception))


class EconomyChecks(unittest.TestCase):
    def test_spending_caps_share_across_retries_and_include_truncated_requests(self):
        from vision_costs import SpendLedger,VisionBudgetError
        ledger=SpendLedger(session_limit=1.,move_limit=.10,move_calls=2)
        move=SimpleNamespace()
        for _ in range(2):
            t=ledger.reserve(move,'gpt-6.1-sol','default',1024)
            row=ledger.finish(t,dict(input_tokens=1600,output_tokens=1024))
        self.assertAlmostEqual(move._vision_spend['spent'],2*(.0032+.01024))
        with self.assertRaises(VisionBudgetError):ledger.reserve(move,'gpt-6.1-sol','default',1024)
        # A new move gets a separate move allowance, but launch spending persists.
        t=ledger.reserve(SimpleNamespace(),'gpt-6.1-sol','default',1024)
        ledger.finish(t,dict(input_tokens=1600,output_tokens=100))
        self.assertGreater(ledger.spent,move._vision_spend['spent'])
        tight=SpendLedger(session_limit=.001)
        with self.assertRaises(VisionBudgetError):tight.reserve(SimpleNamespace(),'gpt-6.1-sol','default',1024)

    def test_budget_rejection_never_calls_the_api(self):
        sk=bare_skill();sk.model='gpt-6.1-sol';sk.effort='low'
        sk._cost_ledger=A.SpendLedger(session_limit=.001)
        with patch.object(sk,'_post') as post,self.assertRaises(RuntimeError):
            sk._model_answer('look',CROP)
        post.assert_not_called()

    def test_cheap_defaults_and_launcher_use_scripted_place(self):
        sk=A.AstraSkill(repo='astra:mock')
        self.assertEqual(sk.model,'gpt-6.1-terra')
        self.assertEqual(sk.effort,'low')
        self.assertEqual(sk.service_tier,'default')
        launcher=(HERE.parent/'scripts'/'play_astra.sh').read_text()
        self.assertIn('MODEL=${ASTRA_MODEL:-gpt-6.1-terra}',launcher)
        self.assertIn('--model-repo "scripted:place"',launcher)
        self.assertIn('--grasp-repo "astra:$MODEL"',launcher)


class ScriptedPlacementChecks(unittest.TestCase):
    def test_unreachable_placement_keeps_grip_closed(self):
        from targets import Projector
        import staged as S
        from scripted_place import ScriptedPlacePolicy
        sk=ScriptedPlacePolicy();q=np.array([20.,-40.,35.,90.,20.,4.])
        sk.goal={'which':('to',)}
        sk.staged=SimpleNamespace(Pl=S.Planner(Projector()),close_h=.02,to_xyz=np.array([.15,0,.01]))
        with patch.object(sk.staged.Pl,'ik',return_value=None):
            action=sk.act(q,CROP,'place')
        self.assertIsNotNone(sk.gave_up)
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)

    def test_full_moves_use_vision_only_for_grasp_and_script_for_carry_and_place(self):
        from types import SimpleNamespace
        from targets import Projector,LATEST
        from fake_arm import FakeArm
        from policy_player import SkillDriver,READY6
        import staged as S
        for piece,frm,to in [('pawn','e7','e5'),('knight','g8','f6'),('queen','d8','g4'),('bishop','c8','h3'),('bishop','f1','drop zone')]:
            with self.subTest(piece=piece):
                P=Projector();Pl=S.Planner(P)
                arm=FakeArm(Pl,{frm:(piece,P.square_xyz(frm,LATEST))},READY6)
                task=f'move the black {piece} from {frm} to {to}'
                p=SimpleNamespace(policy=load_policy(repo='scripted:place'),grasp_policy=load_policy(repo='astra:mock'),
                                  model_repo='scripted:place',grasp_repo='astra:mock',async_exec=False,robot=arm,staged=None)
                driver=SkillDriver(p,task)
                mv=S.StagedMove(Pl,driver,task,LATEST,READY6,0.,drop_xyz=P.drop_xyz(LATEST));p.staged=mv
                grasp=p.grasp_policy
                grasp.truth_xy=lambda kind:arm.pieces[frm][1][:2].copy()
                original=grasp._mock_answer
                def truth_answer(q,meta,kind):
                    answer=original(q,meta,kind);answer['grid_px']=[];return answer
                grasp._mock_answer=truth_answer
                def sync_look(q,img,kind):
                    ans,meta,crop,text=grasp._ask(q,img,kind)
                    grasp.pending=dict(done=True,ans=ans,meta=meta,crop=crop,text=text,q=q.copy())
                grasp._start_look=sync_look;grasp._log_look=lambda *args:None
                clock=[0.]
                with patch.object(L.time,'monotonic',side_effect=lambda:clock[0]), \
                     patch.object(p.policy,'_ask',side_effect=AssertionError('placement must never use vision')):
                    while mv.result is None and clock[0]<180:
                        action=mv.step(arm.q,arm.image,clock[0])
                        if action is not None:arm.target=np.clip(action,arm.q-4.,arm.q+4.)
                        arm.tick(1/30);clock[0]+=1/30
                error=1000*np.linalg.norm(arm.pieces[frm][1][:2]-mv.to_xyz[:2])
                self.assertIsNotNone(mv.result)
                self.assertEqual(mv.result[0],'placed')
                self.assertLess(error,5.)
                self.assertEqual(p.policy.cost_calls,0)
                print(f'ECONOMY {piece}: {mv.result}, {error:.2f} mm, {clock[0]:.1f}s, placement API calls 0')


class DropZoneReleaseChecks(unittest.TestCase):
    def recorded_drop(self, drop_zone=True):
        from scripted_place import ScriptedPlacePolicy
        row=json.loads((HERE/'fixtures/scripted_drop_zone_20261110.json').read_text())
        sk=ScriptedPlacePolicy();q=np.array(row['q']);target=np.array(row['to_xyz'])
        sk.staged=SimpleNamespace(close_h=row['close_h_reconstructed'],to_xyz=target,
                                  ring=target.copy() if drop_zone else None)
        sk.goal={'which':('to',)};sk.phase='settling';sk.desired=np.array(row['command'])
        sk.grip=L.CLOSE_GRIP;sk.bias=np.array([0.,-5.,0.,0.,0.,0.])
        sk._release_target=target.copy();sk._release_target[2]+=sk.staged.close_h+.003
        sk._settle_deadline=108.;sk.settle_start=100.;sk.settle_t0=None
        return sk,q

    def tick(self,sk,q,now):
        with patch.object(L.time,'monotonic',return_value=now):return sk.act(q,CROP,'put bishop in drop zone')

    def test_recorded_reached_drop_zone_releases_despite_precise_height_error(self):
        sk,q=self.recorded_drop();position,_=sk.Pl.fk(q)
        self.assertLess(np.linalg.norm(position[:2]-sk.staged.to_xyz[:2]),.002)
        self.assertGreater(abs(position[2]-sk._release_target[2]),.002)
        action=self.tick(sk,q,100.)
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
        np.testing.assert_allclose(action[:5],q[:5],atol=1e-4)
        action=self.tick(sk,q,100.35)
        self.assertEqual(sk.phase,'released')
        self.assertIsNone(sk.gave_up)
        self.assertAlmostEqual(float(action[5]),L.RELEASE_GRIP)
        np.testing.assert_allclose(action[:5],q[:5],atol=1e-4)
        np.testing.assert_array_equal(sk.bias,np.zeros(6))

    def test_square_placement_keeps_its_precise_height_check(self):
        sk,q=self.recorded_drop(drop_zone=False)
        self.tick(sk,q,100.);action=self.tick(sk,q,102.)
        self.assertNotEqual(sk.phase,'released')
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)

    def test_drop_zone_does_not_release_outside_area_or_high_above_surface(self):
        for high in (False,True):
            with self.subTest(high=high):
                sk,q=self.recorded_drop()
                if high:sk.staged.close_h=.001  # inferred base is >15 mm above the surface
                else:sk._release_target[0]+=.03
                self.tick(sk,q,100.);action=self.tick(sk,q,101.)
                self.assertNotEqual(sk.phase,'released')
                self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)

    def test_drop_zone_waits_for_physical_motion_to_stop(self):
        sk,q=self.recorded_drop();self.tick(sk,q,100.)
        moving=q.copy();moving[0]+=.5
        self.tick(sk,moving,100.2)
        action=self.tick(sk,moving,100.4)
        self.assertAlmostEqual(float(action[5]),L.CLOSE_GRIP)
        action=self.tick(sk,moving,100.6)
        self.assertAlmostEqual(float(action[5]),L.RELEASE_GRIP)


class TwinChecks(unittest.TestCase):
    def test_full_grasp_carry_place_and_return_for_both_backends(self):
        import staged as S
        from fake_arm import FakeArm
        from policy_player import SkillDriver, READY6
        from targets import Projector, LATEST

        # No rendering needed: astra:mock / claude:mock point using the twin's truth.
        results = {}
        for provider in ('astra', 'claude'):
            for piece, frm, to in [('pawn', 'e7', 'e5'), ('knight', 'g8', 'f6'), ('queen', 'd8', 'g4'), ('bishop', 'c8', 'h3')]:
                with self.subTest(provider=provider, piece=piece):
                    P = Projector(); Pl = S.Planner(P)
                    xyz = P.square_xyz(frm, LATEST)
                    arm = FakeArm(Pl, {frm: (piece, xyz)}, READY6)
                    task = f'move the black {piece} from {frm} to {to}'
                    p = SimpleNamespace(policy=load_policy('final', 'skills', f'{provider}:mock'),
                                        grasp_policy=load_policy('final', 'skills', f'{provider}:mock'),
                                        model_repo=f'{provider}:mock', grasp_repo=f'{provider}:mock',
                                        async_exec=False, robot=arm, staged=None)
                    self.assertEqual(p.policy.type, provider)
                    if provider == 'astra':
                        self.assertEqual(p.policy.model, 'gpt-6.1-terra')
                    driver = SkillDriver(p, task)
                    mv = S.StagedMove(Pl, driver, task, LATEST, READY6, 0.0, drop_xyz=P.drop_xyz(LATEST))
                    p.staged = mv
                    for sk in (p.policy, p.grasp_policy):
                        # Isolate controller geometry from the fake scene's nominal board size.
                        # The real session includes fitted size/correction maps the rigid fake board lacks.
                        sk.truth_xy = lambda kind: (arm.pieces[frm][1][:2].copy() if kind == "from" else mv.to_xyz[:2].copy())
                        mock_answer = sk._mock_answer
                        def truth_answer(q, meta, kind, original=mock_answer):
                            ans = original(q, meta, kind)
                            ans["grid_px"] = []
                            return ans
                        sk._mock_answer = truth_answer
                        def sync_look(q, img, kind, sk=sk):
                            ans, meta, crop, text = sk._ask(q, img, kind)
                            sk.pending = dict(done=True, ans=ans, meta=meta, crop=crop, text=text, q=q.copy())
                        sk._start_look = sync_look
                        sk._log_look = lambda *args: None
                    clock = [0.0]
                    with patch.object(L.time, 'monotonic', side_effect=lambda: clock[0]):
                        while mv.result is None and clock[0] < 180:
                            action = mv.step(arm.q, arm.image, clock[0])
                            if action is not None:
                                arm.target = np.clip(action, arm.q - 4.0, arm.q + 4.0)
                            arm.tick(1 / 30)
                            clock[0] += 1 / 30
                    error_mm = 1000 * np.linalg.norm(arm.pieces[frm][1][:2] - mv.to_xyz[:2])
                    results[provider, piece] = (mv.result, error_mm)
                    print(f'TWIN {provider} {piece}: {mv.result}, placement error {error_mm:.2f} mm, {clock[0]:.1f} simulated seconds')
                    self.assertIsNotNone(mv.result)
                    self.assertEqual(mv.result[0], 'placed')
                    self.assertIsNone(arm.held)
                    self.assertLess(error_mm, 5.0)
        for piece in ('pawn', 'knight', 'queen', 'bishop'):
            self.assertEqual(results['astra', piece][0], results['claude', piece][0])
            self.assertAlmostEqual(results['astra', piece][1], results['claude', piece][1], delta=.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)

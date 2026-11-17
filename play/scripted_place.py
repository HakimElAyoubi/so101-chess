"""Calibrated place-only policy: no camera inference or API calls."""
import math
import time
import numpy as np
from llm_skill import ClaudeSkill, CLOSE_GRIP, RELEASE_GRIP


class ScriptedPlacePolicy(ClaudeSkill):
    type = 'scripted'
    model_default = 'place'

    def __init__(self, checkpoint='final', variant='skills', repo=None):
        super().__init__(checkpoint,variant,repo='scripted:place')
        self.timeout_s = 20.

    def reset(self):
        super().reset()
        self._waypoints = []
        self._release_target = None
        self._settle_deadline = None
        self._drop_stable_pose = None
        self._drop_stable_since = None

    def act(self, state6, image, task):
        q=np.asarray(state6,float);now=time.monotonic()
        if self.staged is None:  # loader warm-up, never an open command
            return q.copy()
        if 'to' not in (self.goal or {}).get('which',()):
            self.gave_up='scripted placement is place-only';return q.copy()
        if self.phase == 'init':
            self.Pl=self.staged.Pl
            self.desired=q.copy();self.bias=np.zeros(6);self.grip=CLOSE_GRIP
            if not self.staged.close_h or self.staged.close_h <= 0:
                self.gave_up='missing measured holding height';return self._command(q)
            start,_=self.Pl.fk(q)
            target=np.asarray(self.staged.to_xyz,float).copy()
            target[2]+=self.staged.close_h+.003
            tilt,psi=self.Pl.orientation(q)
            seed=q.copy();seed[5]=CLOSE_GRIP
            path=[]
            for f in np.linspace(0,1,max(2,math.ceil(np.linalg.norm(target-start)/.003)+1))[1:]:
                point=start+f*(target-start)
                candidate=self.Pl.ik(point,tilt,psi,seed,CLOSE_GRIP)
                if candidate is None or np.linalg.norm(self.Pl.fk(candidate)[0]-point)>.002:
                    self.gave_up='scripted place descent is unreachable; keeping grip closed'
                    return self._command(q)
                path.append(candidate);seed=candidate
            self._waypoints=path;self._release_target=target
            self.phase='lowering'
        if self.gave_up:
            return self._command(q)
        if self.phase == 'lowering':
            if self._moving(now):
                return self._command(q)
            if self._waypoints:
                candidate=self._waypoints.pop(0)
                duration=max(.15,1.875*float(np.abs(candidate[:5]-self.desired[:5]).max())/30.)
                self._begin_move(candidate,now,duration)
                return self._command(q)
            self.phase='settling';self.settle_start=now;self.settle_t0=None
            self._settle_deadline=now+8.
        if self.phase == 'settling':
            position,_=self.Pl.fk(q)
            error=position-self._release_target
            if getattr(self.staged,'ring',None) is not None:
                # A capture goes into an area, not onto a precise board square.
                # Use the measured pose and inferred piece-base height so contact
                # with the table cannot prevent opening indefinitely.
                base_above_surface=position[2]-self.staged.close_h-self.staged.to_xyz[2]
                at_target=np.linalg.norm(error[:2])<=.015 and -.005<=base_above_surface<=.015
                if at_target:
                    self.desired=q.copy();self.bias=np.zeros(6)
                if not at_target:
                    self._drop_stable_pose=self._drop_stable_since=None
                elif (self._drop_stable_pose is None or
                      np.max(np.abs(q[:5]-self._drop_stable_pose[:5]))>.3):
                    self._drop_stable_pose=q.copy();self._drop_stable_since=now
                ready=at_target and self._drop_stable_since is not None and now-self._drop_stable_since>=.3
            else:
                at_target=np.linalg.norm(error[:2])<=.003 and abs(error[2])<=.002
                ready=at_target and self._settled(q,now)
            if ready:
                # Hold the reached physical pose while opening, rather than
                # continuing to integrate against an unreachable height target.
                self.desired=q.copy();self.bias=np.zeros(6)
                self.grip=RELEASE_GRIP;self.phase='released'
            elif now >= self._settle_deadline:
                self.gave_up='scripted place did not reach target; keeping grip closed'
        return self._command(q)

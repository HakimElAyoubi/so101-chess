"""Local ready-pose before/after evidence for an expected human chess move.

Checks image changes at the actual source and destination, plus scene stability.
This is change detection, not general chess-piece recognition. No network calls.
"""
import cv2
import numpy as np
import chess

HEIGHTS={'pawn':.025,'knight':.031,'bishop':.036,'rook':.028,'queen':.040,'king':.050}


class HumanMoveWatch:
    def __init__(self, projector, session, board, move, image, joints):
        self.q=np.asarray(joints,float).copy();self.before=self.gray(image)
        self.previous=self.before.copy();self.stable_since=None
        self.source=chess.square_name(move.from_square);self.dest=chess.square_name(move.to_square)
        moving=board.piece_at(move.from_square)
        points=[]
        for sq in chess.SQUARES:
            piece=board.piece_at(sq) or moving
            h=HEIGHTS[chess.piece_name(piece.piece_type)]
            xyz=np.asarray(projector.square_xyz(chess.square_name(sq),session),float)
            points.extend(xyz+[x,y,z] for z in (.003,h+.003) for x,y in ((-.007,-.007),(.007,-.007),(.007,.007),(-.007,.007)))
        pixels,visible=projector.project(self.q,np.asarray(points),image.shape[1])
        self.masks={};self.board_mask=np.zeros(self.before.shape,np.uint8)
        for sq in chess.SQUARES:
            p=np.asarray(pixels[sq*8:(sq+1)*8])
            if not np.isfinite(p).all():continue
            # At ready some tall tops extend beyond the frame, while their
            # lower body/base remains clearly visible. Use that visible portion.
            hull=cv2.convexHull(p.astype(np.int32))
            mask=np.zeros(self.before.shape,np.uint8)
            cv2.fillConvexPoly(mask,hull,255)
            area=cv2.countNonZero(mask)
            if area<60 or area/max(1.,cv2.contourArea(hull))<.30:continue
            self.masks[chess.square_name(sq)]=mask.astype(bool)
            self.board_mask|=mask
        if self.source not in self.masks or self.dest not in self.masks:
            raise ValueError('The source/destination is outside the ready camera view. Check board calibration.')
        self.board_mask=self.board_mask.astype(bool)
        self.last=dict(confirmed=False,message='Waiting for your move.',source=self.source,destination=self.dest)

    @staticmethod
    def gray(image):
        return cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(5,5),0)

    def observe(self,image,joints,now):
        if np.max(np.abs(np.asarray(joints)[:5]-self.q[:5]))>1.:
            self.stable_since=None
            self.last=dict(confirmed=False,message='Arm moved from its camera baseline. Return to ready before checking.')
            return self.last
        gray=self.gray(image)
        if gray.shape!=self.before.shape:
            self.last=dict(confirmed=False,message='Camera dimensions changed. Restart this game.');return self.last
        # Remove a uniform exposure shift, but do not register away a piece move.
        corrected=np.clip(gray.astype(float)-np.median(gray.astype(float)-self.before),0,255)
        difference=np.abs(corrected-self.before)
        scores={sq:float(np.mean(difference[mask]>20)) for sq,mask in self.masks.items()}
        motion=float(np.mean(np.abs(gray.astype(float)-self.previous)[self.board_mask]))
        self.previous=gray
        changed=sorted(scores,key=scores.get,reverse=True)
        widespread=sum(v>.18 for v in scores.values())>6
        evidence=scores[self.source]>.10 and scores[self.dest]>.10 and not widespread
        if evidence and motion<1.5:
            if self.stable_since is None:self.stable_since=now
        else:self.stable_since=None
        confirmed=self.stable_since is not None and now-self.stable_since>=1.
        message=('Move seen; board is stable.' if confirmed else
                 'Remove your hand and leave the board still.' if widespread or motion>=1.5 else
                 'Expected source and destination have not both changed.' if not evidence else 'Checking that the board stays still…')
        self.last=dict(confirmed=confirmed,message=message,source=self.source,destination=self.dest,
                       source_change=round(scores[self.source],3),destination_change=round(scores[self.dest],3),
                       changed_squares=changed[:4],motion=round(motion,2))
        return self.last

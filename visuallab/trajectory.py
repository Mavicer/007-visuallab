"""A short-lived display trail, separate from detection and control outputs.

This is local visual association, not long-term target identity or a gimbal
tracker. Only observed centers are stored; missing motion is never invented.
"""
from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .detection import Observation


@dataclass(frozen=True)
class TrailPoint:
    center: tuple[int, int]
    time: float
    start: bool  # Lift the pen after a missing frame; do not bridge a gap.


def _overlap(a: Observation, b: Observation) -> float:
    ax,ay,aw,ah = a.box
    bx,by,bw,bh = b.box
    intersection = max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection/max(1,min(aw*ah,bw*bh))


class TrajectoryTrail:
    """One confirmed subject -> cyan fading trail and a pulsing head ring."""

    def __init__(self, *, labels: frozenset[str] = frozenset({'+'}),
                 enabled: bool = True, lifetime: float = 1.5) -> None:
        self.labels = labels
        self.enabled = enabled
        self.lifetime = lifetime
        self.points: deque[TrailPoint] = deque(maxlen=128)
        self.reset()

    def reset(self) -> None:
        self.points.clear()
        self.target: Observation | None = None
        self.last_seen = self.confirmed_at = float('-inf')
        self.last_update = None
        self.shape = None
        self.pen_up = True
        self.visible = False

    def update(self, observations: list[Observation], shape: tuple[int,...], now: float,
               *, associated: bool = False) -> None:
        size = shape[:2]
        if self.shape != size or (self.last_update is not None and now < self.last_update):
            self.reset()
        self.shape,self.last_update,self.visible = size,now,False
        if not self.enabled:
            self.reset()
            return
        while self.points and now-self.points[0].time > self.lifetime:
            self.points.popleft()
        if now-self.last_seen > .55:
            self.target = None
        h,w = size
        valid = [o for o in observations if 0<=o.center[0]<w and 0<=o.center[1]<h
                 and o.box[2]>0 and o.box[3]>0]
        confirmed = [o for o in valid if not o.is_unknown and o.label in self.labels]
        # Overlapping duplicate OCR boxes should not look like two subjects.
        distinct = []
        for o in sorted(confirmed,key=lambda o:o.score if o.score is not None else 0.,reverse=True):
            if not any(_overlap(o,previous) > .7 for previous in distinct):
                distinct.append(o)
        if associated:
            # A real tracker has already assigned the single identity. Do not
            # apply a second, tighter display-only motion gate on fast motion.
            if len(distinct) != 1:
                self.pen_up = True
                return
            selected = distinct[0]
        elif self.target is None:
            if len(distinct) != 1:
                self.pen_up = True
                return
            selected = distinct[0]
            self.points.clear()  # Reacquisition begins a fresh visual history.
            self.pen_up = True
        else:
            diagonal = math.hypot(w,h)
            radius = min(.22*diagonal,max(.08*diagonal,2*math.sqrt(self.target.box[2]*self.target.box[3])))
            def near(o):
                return math.dist(o.center,self.target.center) <= radius
            options = [o for o in distinct if near(o)]
            if not options and not confirmed and now-self.confirmed_at <= .4:
                # Brief OCR dropout may use an overlapping *unknown* box.
                # A different accepted character never becomes a '+' target.
                options = [o for o in valid if o.is_unknown and near(o) and _overlap(o,self.target)>=.2]
            if len(options) != 1:
                self.pen_up = True
                return
            selected = options[0]
        self.target,self.last_seen,self.visible = selected,now,True
        if not selected.is_unknown:
            self.confirmed_at = now
        self.points.append(TrailPoint(selected.center,now,self.pen_up))
        self.pen_up = False

    def draw(self, marked: np.ndarray, now: float) -> None:
        """Modify only the caller's color display copy, using a bounded ROI."""
        if not self.enabled:
            return
        points = [p for p in self.points if 0<=now-p.time<=self.lifetime]
        if not points:
            return
        h,w = marked.shape[:2]
        width = max(5,round(min(h,w)/100))
        # Deterministic, time-based sparks are display decoration. They never
        # enter observations, saved raw frames, or a future control interface.
        sparks = []
        for p in points[-32:]:
            age = now-p.time
            if age > .55 or p.start:
                continue
            seed = int(p.time*1000)+p.center[0]*37+p.center[1]*17
            for j in range(3):
                angle = ((seed*17+j*137)%360)*math.pi/180
                speed = (35+(seed+j*23)%80)*max(1.,width/5)
                position = (round(p.center[0]+math.cos(angle)*speed*age),
                            round(p.center[1]+math.sin(angle)*speed*age+45*age*age))
                sparks.append((position,age))
        coords = np.asarray([p.center for p in points]+[p for p,a in sparks],np.int32)
        pad = max(24,width*3)
        left,top = np.maximum(coords.min(axis=0)-pad,0)
        right,bottom = np.minimum(coords.max(axis=0)+pad+1,[w,h])
        if right<=left or bottom<=top:
            return
        # Two small additive layers give a soft glow without blurring or
        # recoloring the camera image, nor blending a full frame per segment.
        glow = np.zeros((bottom-top,right-left,3),np.uint8)
        sharp = np.zeros_like(glow)
        offset = np.array([left,top])
        for previous,current in zip(points,points[1:]):
            if current.start:
                continue
            fade = max(0.,1-(now-previous.time)/self.lifetime)**1.5
            color = tuple(round(channel*fade) for channel in (255,185,35))
            start,end = tuple(np.asarray(previous.center)-offset),tuple(np.asarray(current.center)-offset)
            cv2.line(glow,start,end,color,width*3,cv2.LINE_AA)
            cv2.line(sharp,start,end,color,width,cv2.LINE_AA)
            core = tuple(round(channel*fade) for channel in (230,255,255))
            cv2.line(sharp,start,end,core,max(1,width//3),cv2.LINE_AA)
        for position,age in sparks:
            fade = (1-age/.55)**2
            center = tuple(np.asarray(position)-offset)
            color = tuple(round(c*fade) for c in (255,210,65))
            cv2.circle(glow,center,max(2,width//2),color,-1,cv2.LINE_AA)
            cv2.circle(sharp,center,max(1,width//5),color,-1,cv2.LINE_AA)
        if self.visible and now-self.last_seen < .12:
            center = tuple(np.asarray(points[-1].center)-offset)
            radius = width+7+round(2*math.sin(now*6))
            cv2.circle(glow,center,radius,(255,185,35),4,cv2.LINE_AA)
            cv2.circle(sharp,center,radius,(255,220,100),1,cv2.LINE_AA)
            cv2.circle(sharp,center,3,(235,255,255),-1,cv2.LINE_AA)
        roi = marked[top:bottom,left:right]
        blurred = cv2.GaussianBlur(glow,(9,9),0)
        cv2.addWeighted(roi,1.,blurred,.4,0,dst=roi)
        cv2.addWeighted(roi,1.,sharp,.85,0,dst=roi)

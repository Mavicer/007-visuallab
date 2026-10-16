"""Generic single-object association with short, image-measured LK fallback.

The detector still runs on every frame. Optical flow bridges only a brief
miss, with forward/backward and geometric checks; it never extrapolates a
position through a long occlusion. No OCR or marker color is required here.
"""
import math
from time import perf_counter

import cv2
import numpy as np

from .detection import DetectionResult, FrameDetector, Observation


class TrackedObjectDetector:
    def __init__(self, detector: FrameDetector, *, label: str = 'armor',
                 flow_seconds: float = .18, lost_seconds: float = .5) -> None:
        if not label or not 0 <= flow_seconds <= lost_seconds:
            raise ValueError('Invalid tracking label or time limits')
        names = getattr(detector, 'class_names', None)
        if isinstance(names, dict):
            names = names.values()
        if names is not None and label not in names:
            raise ValueError(f'权重缺少 {label!r} 类；请使用整板定位权重')
        self.detector, self.label = detector, label
        self.flow_seconds, self.lost_seconds = flow_seconds, lost_seconds
        self.identity = 0
        self.reset()

    def reset(self) -> None:
        self.target = None
        self.previous = None
        self.previous_time = None
        self.last_detection = float('-inf')
        self.flow_frames = 0
        self.size = None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        return self.detect_at(frame, perf_counter())

    def detect_at(self, frame: np.ndarray, now: float) -> DetectionResult:
        """Explicit monotonic time also permits honest timestamped replay."""
        h, w = frame.shape[:2]
        if (self.size != (h, w) or (self.previous_time is not None and
                (now < self.previous_time or now-self.previous_time > self.lost_seconds))):
            self.reset()
        self.size = (h, w)
        if now-self.last_detection > self.lost_seconds:
            self.target = None
        raw = self.detector.detect(frame)
        candidates = [o for o in raw.observations if o.label == self.label and
                      not o.is_unknown and o.box[2] > 0 and o.box[3] > 0 and
                      0 <= o.center[0] < w and 0 <= o.center[1] < h]
        # NMS is normally provided by YOLO. Suppress overlapping duplicates
        # from other pluggable detectors before testing ambiguity.
        distinct = []
        for o in sorted(candidates, key=lambda o: o.score or 0., reverse=True):
            if not any(self._overlap(o, other) > .7 for other in distinct):
                distinct.append(o)
        candidates = distinct
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (min(640, w), max(1, round(h*min(640,w)/w))))
        selected, state = None, 'lost'
        if self.target is None:
            if len(candidates) == 1:
                selected = candidates[0]
                self.identity += 1
            elif candidates:
                state = 'ambiguous'
        else:
            # Expand the gate for fast measured motion, without smoothing the
            # center backwards in time. Distant subjects cannot steal a lock.
            radius = min(.3*math.hypot(w,h), max(.12*math.hypot(w,h),
                         2*math.hypot(*self.target.box[2:])))
            near = [o for o in candidates if math.dist(o.center,self.target.center) <= radius
                    and .25 <= o.box[2]*o.box[3]/max(1,self.target.box[2]*self.target.box[3]) <= 4.]
            if len(near) == 1:
                selected = near[0]
            elif near:
                state = 'ambiguous'
            elif not candidates and now-self.last_detection <= self.flow_seconds and self.flow_frames < 3:
                selected = self._flow(small, (h,w))
                if selected is not None:
                    state = 'flow'
                    self.flow_frames += 1
        if selected is not None:
            if state != 'flow':
                state = 'detector'
                self.last_detection, self.flow_frames = now, 0
            self.target = selected
        elif self.target is not None:
            # Once the image evidence fails, do not restart flow from a newer
            # background image using an old target box. Wait for detection.
            self.flow_frames = 3
        self.previous, self.previous_time = small, now
        binary = np.zeros_like(gray)
        if selected is not None:
            x,y,bw,bh = selected.box
            # This is a diagnostic local binary view, not the detector input.
            _,binary[y:y+bh,x:x+bw] = cv2.threshold(gray[y:y+bh,x:x+bw],0,255,
                                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        raw.observations = [selected] if selected is not None else []
        raw.gray, raw.binary, raw.score_name = gray, binary, 'conf'
        raw.diagnostics['tracking'] = dict(state=state, identity=self.identity,
            candidates=len(candidates), detection_age_ms=round(max(0.,now-self.last_detection)*1000,1)
            if self.last_detection != float('-inf') else None)
        return raw

    @staticmethod
    def _overlap(a, b):
        x,y,w,h=a.box;xx,yy,ww,hh=b.box
        return max(0,min(x+w,xx+ww)-max(x,xx))*max(0,min(y+h,yy+hh)-max(y,yy))/max(1,min(w*h,ww*hh))

    def _flow(self, current: np.ndarray, original_size: tuple[int,int]) -> Observation | None:
        if self.previous is None or self.target is None:
            return None
        h,w=original_size;sy,sx=current.shape[0]/h,current.shape[1]/w
        x,y,bw,bh=self.target.box
        mask=np.zeros_like(current)
        # Stay inside the last box: surrounding faces/hands are poor features.
        x1,y1=round((x+.08*bw)*sx),round((y+.08*bh)*sy)
        x2,y2=round((x+.92*bw)*sx),round((y+.92*bh)*sy)
        mask[max(0,y1):min(current.shape[0],y2),max(0,x1):min(current.shape[1],x2)]=255
        points=cv2.goodFeaturesToTrack(self.previous,60,.015,3,mask=mask)
        if points is None or len(points)<6:
            return None
        params=dict(winSize=(21,21),maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.02))
        forward,status,_=cv2.calcOpticalFlowPyrLK(self.previous,current,points,None,**params)
        if forward is None or status is None:return None
        backward,back_status,_=cv2.calcOpticalFlowPyrLK(current,self.previous,forward,None,**params)
        if backward is None or back_status is None:return None
        good=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(points-backward,axis=2).ravel()<1.5)
        before,after=points[good].reshape(-1,2),forward[good].reshape(-1,2)
        if len(before)<6 or len(before)<.5*len(points):return None
        matrix,inliers=cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.)
        if matrix is None or inliers is None or inliers.sum()<6 or inliers.mean()<.7:return None
        retained=before[inliers.ravel()==1]
        if np.any(np.ptp(retained,axis=0)<np.array([bw*sx,bh*sy])*.18):return None
        scale=math.hypot(matrix[0,0],matrix[1,0])
        if not .8<=scale<=1.25 or abs(math.atan2(matrix[1,0],matrix[0,0]))>math.radians(35):return None
        corners=np.array([[x*sx,y*sy],[(x+bw)*sx,y*sy],[(x+bw)*sx,(y+bh)*sy],[x*sx,(y+bh)*sy]],np.float32)
        mapped=cv2.transform(corners[None],matrix)[0]/[sx,sy]
        x1,y1=np.floor(mapped.min(axis=0)).astype(int);x2,y2=np.ceil(mapped.max(axis=0)).astype(int)
        # Reject clipped or runaway flow instead of clamping it into a fake box.
        if x1<0 or y1<0 or x2>w or y2>h or x2<=x1 or y2<=y1:return None
        if math.dist(((x1+x2)/2,(y1+y2)/2),self.target.center)>.18*math.hypot(w,h):return None
        # Identity can persist briefly; optical flow has no model confidence.
        return Observation((int(x1),int(y1),int(x2-x1),int(y2-y1)),
                           (int((x1+x2)//2),int((y1+y2)//2)),self.label,None)

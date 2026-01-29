import numpy as np
from copy import deepcopy

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0.0

class Track:
    def __init__(self, tid, box):
        self.id = tid
        self.box = box
        self.lost = 0

class IOUTracker:
    def __init__(self, max_lost=15, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 0
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2]
        returns list of dict {'id':id, 'box':box}
        """
        assigned_tracks = set()
        results = []
        if len(self.tracks)==0:
            for d in detections:
                t = Track(self.next_id, d)
                self.next_id += 1
                self.tracks.append(t)
                results.append({"id":t.id, "box":t.box})
            return results

        # compute iou matrix
        ious = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for i,t in enumerate(self.tracks):
            for j,d in enumerate(detections):
                ious[i,j]=iou(t.box, d)

        # greedy assign
        for _ in range(min(len(self.tracks), len(detections))):
            i,j = np.unravel_index(np.argmax(ious), ious.shape)
            if ious[i,j] < self.iou_threshold:
                break
            self.tracks[i].box = detections[j]
            self.tracks[i].lost = 0
            results.append({"id":self.tracks[i].id, "box":self.tracks[i].box})
            ious[i,:] = -1
            ious[:,j] = -1
            assigned_tracks.add(i)

        # mark unassigned tracks
        for idx, t in enumerate(self.tracks):
            if idx not in assigned_tracks:
                t.lost += 1

        # create new tracks for unassigned detections
        assigned_dets = set()
        for r in results:
            # find index of detection equal to box (approx)
            for idx,d in enumerate(detections):
                if d == r["box"]:
                    assigned_dets.add(idx)
                    break
        for idx,d in enumerate(detections):
            if idx not in assigned_dets:
                t = Track(self.next_id, d)
                self.next_id += 1
                self.tracks.append(t)
                results.append({"id":t.id, "box":t.box})

        # remove lost tracks
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]

        return results
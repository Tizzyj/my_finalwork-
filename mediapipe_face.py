import cv2
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

class MediaPipeFace:
    def __init__(self, min_detection_confidence=0.5):
        self.det = mp_face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)
        self.mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=8)

    def detect(self, frame):
        """Returns list of faces: {'box':[x1,y1,x2,y2], 'landmarks':..., 'crop':crop_image}"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.det.process(rgb)
        faces = []
        if res.detections:
            for d in res.detections:
                # bounding box
                bb = d.location_data.relative_bounding_box
                x1 = int(max(0, bb.xmin * w))
                y1 = int(max(0, bb.ymin * h))
                x2 = int(min(w, (bb.xmin + bb.width) * w))
                y2 = int(min(h, (bb.ymin + bb.height) * h))
                crop = frame[y1:y2, x1:x2].copy()
                faces.append({"box":[x1,y1,x2,y2], "landmarks":None, "crop":crop})
        return faces
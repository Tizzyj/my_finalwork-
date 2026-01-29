import cv2

def draw_overlay(frame, results):
    """results: dict tid -> {'box':[x1,y1,x2,y2], 'label':label, 'conf':conf}"""
    for tid, info in results.items():
        x1,y1,x2,y2 = [int(i) for i in info["box"]]
        label = info["label"]
        conf = info["conf"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        text = f"id:{tid} {label} {conf:.2f}"
        cv2.putText(frame, text, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
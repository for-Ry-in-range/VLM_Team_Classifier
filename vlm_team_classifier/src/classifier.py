import cv2
from ultralytics import YOLO

def main(video_path):
    # Initializations
    detector = YOLO('yolov8n.pt') 
    classifier = SigLIPTeamClassifier()
    cap = cv2.VideoCapture(video_path)
    
    # Fit on the fist frame that have at least 5 players
    print("Looking for a frame to create the team clusters...")
    while True:
        res, frame = cap.read()
        if not res:
            break
        
        # YOLO inference
        detected = detector(frame, verbose=False)[0]  # verbose=False makes YOLO work silently
        
        # Get indices of people (id of 0) detected
        person_indices = [i for i, c in enumerate(detected.boxes.cls) if int(c) == 0]
        
        if len(person_indices) >= 5:
            initial_bboxes = detected.boxes.xyxy[person_indices].cpu().numpy()
            classifier.fit(frame, initial_bboxes)
            break
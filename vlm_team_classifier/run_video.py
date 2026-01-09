import cv2
from ultralytics import YOLO
from src.classifier import SigLIPTeamClassifier


def main(video_path):
    # Initializations
    detector = YOLO('yolov8n.pt') 
    classifier = SigLIPTeamClassifier()
    vid_cap = cv2.VideoCapture(video_path)
    
    # Fit on the first frame that have at least 9 people
    print("Looking for a frame to create the clusters...")
    while True:
        res, frame = vid_cap.read()
        if not res:
            break
        
        # YOLO inference
        detected = detector(frame, verbose=False)[0]  # verbose=False makes YOLO work silently
        
        # Get indices of people (id of 0) detected
        person_indices = [i for i, c in enumerate(detected.boxes.cls) if int(c) == 0]
        
        if len(person_indices) >= 9:
            initial_bboxes = detected.boxes.xyxy[person_indices].cpu().numpy()
            classifier.fit(frame, initial_bboxes)
            break

    # Team A: blue, Team B: red, Refs: yellow
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 255)]

    while vid_cap.isOpened():
        res, frame = vid_cap.read()
        if not res:
            break
        
        results = detector(frame, verbose=False)[0]
        
        for box in results.boxes:

            # Skip non people
            if int(box.cls) != 0: 
                continue
            
            bbox = box.xyxy[0].cpu().numpy()
            
            # Predict 0, 1, or 2
            cluster_id = classifier.predict(frame, bbox)
            
            if cluster_id == 2:
                label = "Referee"
            elif cluster_id == 1:
                label = "Team B"
            elif cluster_id == 0:
                label = "Team A"
            else:
                label = "Unknown"

            # Draw the boxes
            x1, y1, x2, y2 = map(int, bbox)  # convert bbox values to ints
            if cluster_id >= 0:
                color = colors[cluster_id]
            else:
                color = (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('SigLIP: Teams and Referees', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    vid_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("nba_gameplay.mp4")
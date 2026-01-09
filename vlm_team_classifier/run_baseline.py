import cv2
from ultralytics import YOLO
from baseline import TeamClustering


def is_on_court(bbox, frame_shape):
    """Returns True if the person is a player or ref"""
    x1, y1, x2, y2 = bbox
    frame_h, frame_w, _ = frame_shape
    box_h = y2 - y1
    box_center_y = (y1 + y2) / 2

    # Ignore people whose feet are in the audience
    if y2 < (frame_h * 0.37): 
        return False

    # Ignore people who are small; they're likely in the audience
    if box_h < (frame_h * 0.18):
        return False
        
    return True

def main(video_path):
    # Initializations
    detector = YOLO('yolov8n.pt') 
    classifier = TeamClustering(n_teams=2)  # Baseline only does 2 teams
    vid_cap = cv2.VideoCapture(video_path)
    
    # Fit on the first frame that have at least 9 people
    print("Looking for a frame to create the clusters...")
    while True:
        res, frame = vid_cap.read()
        if not res:
            break
        
        # YOLO inference
        detected = detector(frame, verbose=False)[0]
        
        # Get valid person figures
        valid_bboxes = []
        for box in detected.boxes:
            if int(box.cls) == 0:  # It's a person
                bbox_arr = box.xyxy[0].cpu().numpy()
                if is_on_court(bbox_arr, frame.shape):
                    valid_bboxes.append(bbox_arr)

        if len(valid_bboxes) >= 9:
            classifier.fit(frame, valid_bboxes)
            break

    # Team A: blue, Team B: red
    colors = [(255, 0, 0), (0, 0, 255)]

    while vid_cap.isOpened():
        res, frame = vid_cap.read()
        if not res:
            break
        
        results = detector(frame, verbose=False)[0]
        
        for box in results.boxes:
            # Skip non people
            if int(box.cls) != 0: 
                continue
            
            bbox_arr = box.xyxy[0].cpu().numpy()

            if not is_on_court(bbox_arr, frame.shape):
                continue
            
            # Predict 0 or 1
            team_id = classifier.predict_team(frame, bbox_arr)
            
            if team_id == 0:
                label = "Team A"
            elif team_id == 1:
                label = "Team B"
            else:
                label = "Unknown"

            # Draw the boxes
            x1, y1, x2, y2 = map(int, bbox_arr)  # convert bbox values to ints
            color = colors[team_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Baseline: Teams', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    vid_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("warriors_bucks.mp4")

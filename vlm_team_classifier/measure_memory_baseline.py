import psutil
import os
import tracemalloc


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

def measure_memory_during_execution():
    """Measure memory at different points for baseline classifier"""
    import cv2
    from ultralytics import YOLO
    from src.baseline import TeamClustering
    
    print("Baseline Memory Usage:\n")
    
    # Before loading models
    mem_before = get_memory_usage_mb()
    print(f"Memory before loading models: {mem_before:.2f} MB")
    
    # After loading YOLO
    detector = YOLO('yolov8n.pt')
    mem_after_yolo = get_memory_usage_mb()
    print(f"Memory after loading YOLO: {mem_after_yolo:.2f} MB")
    print(f"YOLO memory usage: {mem_after_yolo - mem_before:.2f} MB\n")
    
    # After loading baseline classifier
    classifier = TeamClustering(n_teams=2)
    mem_after_baseline = get_memory_usage_mb()
    print(f"Memory after loading Baseline: {mem_after_baseline:.2f} MB")
    print(f"Baseline classifier memory usage: {mem_after_baseline - mem_after_yolo:.2f} MB\n")
    
    # Peak memory during processing
    peak_memory = mem_after_baseline
    frame_count = 0
    
    vid_cap = cv2.VideoCapture("warriors_bucks.mp4")
    
    # Memory before fitting
    mem_before_fit = get_memory_usage_mb()
    print(f"Memory before fitting: {mem_before_fit:.2f} MB")
    
    # Find frame for fitting
    while vid_cap.isOpened():
        res, frame = vid_cap.read()
        if not res:
            break
        detected = detector(frame, verbose=False)[0]
        valid_bboxes = []

        for box in detected.boxes:
            if int(box.cls) == 0:  # a person
                bbox_arr = box.xyxy[0].cpu().numpy()
                frame_h = frame.shape[0]
                if bbox_arr[3] >= (frame_h * 0.37) and (bbox_arr[3] - bbox_arr[1]) >= (frame_h * 0.18):
                    valid_bboxes.append(bbox_arr)
        
        if len(valid_bboxes) >= 9:
            classifier.fit(frame, valid_bboxes)
            mem_after_fit = get_memory_usage_mb()
            print(f"Memory after fitting: {mem_after_fit:.2f} MB")
            print(f"Fitting memory usage: {mem_after_fit - mem_before_fit:.2f} MB\n")
            break
    
    # Put video back to start for processing
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    while vid_cap.isOpened():
        res, frame = vid_cap.read()
        if not res:
            break
        
        current_mem = get_memory_usage_mb()
        if current_mem > peak_memory:
            peak_memory = current_mem
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print every 30 frames
            print(f"Frame {frame_count}: Memory = {current_mem:.2f} MB")
    
    vid_cap.release()
    
    print(f"\nPeak memory used: {peak_memory:.2f} MB")
    print(f"Average memory during processing: {get_memory_usage_mb():.2f} MB")

if __name__ == "__main__":
    measure_memory_during_execution()

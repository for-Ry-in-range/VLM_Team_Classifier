import psutil
import os
import tracemalloc


def get_memory_usage_mb():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert bytes to MB

def measure_memory_during_execution():
    """Measure memory at different points"""
    import cv2
    from ultralytics import YOLO
    from src.classifier import SigLIPTeamClassifier
    
    print("Memory Usage:\n")
    
    # Before loading models
    mem_before = get_memory_usage_mb()
    print(f"Memory before loading models: {mem_before:.2f} MB")
    
    # After loading Yolo
    detector = YOLO('yolov8n.pt')
    mem_after_yolo = get_memory_usage_mb()
    print(f"Memory after loading YOLO: {mem_after_yolo:.2f} MB")
    print(f"YOLO memory usage: {mem_after_yolo - mem_before:.2f} MB\n")
    
    # After loading SigLIP
    classifier = SigLIPTeamClassifier()
    mem_after_siglip = get_memory_usage_mb()
    print(f"Memory after loading SigLIP: {mem_after_siglip:.2f} MB")
    print(f"SigLIP memory usage: {mem_after_siglip - mem_after_yolo:.2f} MB\n")
    
    # Highest memory during processing
    peak_memory = mem_after_siglip
    frame_count = 0
    
    vid_cap = cv2.VideoCapture("warriors_bucks.mp4")
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
    
    print(f"Peak memory used: {peak_memory:.2f} MB")
    print(f"Average memory during processing: {get_memory_usage_mb():.2f} MB")

if __name__ == "__main__":
    measure_memory_during_execution()

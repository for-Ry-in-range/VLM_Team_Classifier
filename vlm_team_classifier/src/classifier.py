import torch
import cv2
from collections import Counter
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.cluster import KMeans
import numpy as np


class SigLIPTeamClassifier:
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        # Select GPU (or CPU as last resort)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)

        # Download the weights of the model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()
        
        self.kmeans = None  # No grouping yet
        self.cluster_mapping = None

    def _get_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        
        height = y2 - y1
        
        # Keep the 15% to 60% of the player height
        y2 = int(y1 + (height * 0.6))
        y1 = int(y1 + (height * 0.15))
        
        # Crop the sides
        width = x2 - x1
        x1 = int(x1 + (width * 0.2))
        x2 = int(x2 - (width * 0.2))

        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def get_embedding(self, frame, bbox):
        crop = self._get_crop(frame, bbox)
        if crop is None: 
            return None

        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Process the image into tensors so it can be used by SigLIP model
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():  # no gradient
            outputs = self.model.get_image_features(**inputs)
            
        # Normalize
        siglip_emb = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        siglip_emb = siglip_emb.cpu().numpy().flatten()
        
        # Get the RGB vector
        avg_color = crop.mean(axis=(0, 1))  # returns BGR
        
        # Normalize to 0-1 range so it works with SigLIP
        color_emb = avg_color / 255.0 
        
        # Reorder to RGB
        color_emb = color_emb[::-1] 
        
        # Combine texture and color
        combined_emb = np.concatenate([siglip_emb, color_emb * 1.2])
        
        return combined_emb

    def fit(self, frame, person_bboxes):
        """Finds Team A, Team B, and refs"""
        embeddings = []
        for bbox in person_bboxes:
            emb = self.get_embedding(frame, bbox)
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) < 3:
            print("Not enough people found to cluster.")
            return

        self.kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        cluster_labels = self.kmeans.fit_predict(embeddings)
        
        # Identify which cluster is which based on cluster size
        cluster_counts = Counter(cluster_labels)
        
        # Find the smallest cluster (for refs)
        ref_cluster = min(cluster_counts, key=cluster_counts.get)
        
        team_clusters = [c for c in cluster_counts.keys() if c != ref_cluster]
        
        # Map original cluster id to their team/ref
        self.cluster_mapping = {}
        self.cluster_mapping[ref_cluster] = 2  # Referee
        self.cluster_mapping[team_clusters[0]] = 0  # Team A
        self.cluster_mapping[team_clusters[1]] = 1  # Team B
        
        print("Teams and referees initialized.")

    def predict(self, frame, bbox):
        if self.kmeans is None or self.cluster_mapping is None:
            return -1
        emb = self.get_embedding(frame, bbox)
        if emb is None:
            return -1
        raw_cluster_id = self.kmeans.predict([emb])[0]
        return self.cluster_mapping[raw_cluster_id]
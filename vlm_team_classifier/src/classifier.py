import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.cluster import KMeans


class SigLIPTeamClassifier:
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        # Select GPU (or CPU as last resort)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Download the weights of the model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        self.model.eval()
        
        self.kmeans = None  # No grouping yet

    def _get_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def get_embedding(self, frame, bbox):
        crop = self._get_crop(frame, bbox)
        if crop is None:
            return None
            
        # Convert OpenCV BGR to RGB
        image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # Process the image into tensors so it can be used by SigLIP model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():  # No gradient
            output_embedding = self.model.get_image_features(**inputs)
            
        # Normalize the vector to improve clustering
        embedding = output_embedding / output_embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

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
        self.kmeans.fit(embeddings)
        print("Teams and referees initialized.")

    def predict(self, frame, bbox):
        if self.kmeans is None:
            return -1
        emb = self.get_embedding(frame, bbox)
        if emb is None:
            return -1
        return self.kmeans.predict([emb])[0]
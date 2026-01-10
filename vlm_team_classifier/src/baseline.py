import numpy as np
from sklearn.cluster import KMeans


class TeamClustering:
    """Current baseline: K-Means on jersey colors."""

    def __init__(self, n_teams=2):
        self.n_teams = n_teams
        self.kmeans = None

    def extract_jersey_color(self, frame, bbox):
        """Extract mean RGB color from middle 40% of bbox."""
        x1, y1, x2, y2 = map(int, bbox)
        height = y2 - y1
        margin = int(height * 0.3)
        jersey_region = frame[y1+margin:y2-margin, x1:x2]
        return np.mean(jersey_region.reshape(-1, 3), axis=0)

    def fit(self, frame, player_bboxes):
        """Fit K-Means on initial frame."""
        colors = [self.extract_jersey_color(frame, bbox) for bbox in player_bboxes]
        self.kmeans = KMeans(n_clusters=self.n_teams)
        self.kmeans.fit(colors)
        
    def predict_team(self, frame, bbox):
        """Predict team ID (0 or 1)."""
        color = self.extract_jersey_color(frame, bbox)
        return self.kmeans.predict([color])[0]
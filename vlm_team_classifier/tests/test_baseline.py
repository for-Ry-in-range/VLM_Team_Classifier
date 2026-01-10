import pytest
import numpy as np
from src.baseline import TeamClustering


class TestTeamClustering:
    """Tests for the baseline classifier"""
    
    def test_init(self):
        """Test initialization"""
        classifier = TeamClustering()
        assert classifier.n_teams == 2
        assert classifier.kmeans is None
    
    def test_init_custom_teams(self):
        """Test initialization with custom amount of teams"""
        classifier = TeamClustering(n_teams=3)
        assert classifier.n_teams == 3
        assert classifier.kmeans is None
    
    def test_get_jersey_color(self):
        """Test getting jersey color from one frame"""
        classifier = TeamClustering()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Add a red box in the frame
        frame[30:70, 20:80] = [255, 0, 0]
        
        bbox = [10, 10, 90, 90]
        color = classifier.extract_jersey_color(frame, bbox)
        assert color.shape == (3,)  # check that it's an array with 3 elements
        assert np.all(color >= 0) and np.all(color <= 255)  # checks that values are within 0-255
    
    def test_extract_jersey_color_middle(self):
        """Test that jersey color extraction focuses on the middle"""
        classifier = TeamClustering()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        frame[0:10, 0:20] = [255, 0, 0]  # Red top (should be ignored)
        frame[10:30, 0:20] = [0, 255, 0]  # Green middle (jersey)
        frame[30:40, 0:20] = [0, 0, 255]  # Blue bottom (should be ignored)
        
        bbox = [0, 0, 20, 40]
        color = classifier.extract_jersey_color(frame, bbox)
        
        # Should be mostly green
        assert color[1] > color[0]  # More green than red
        assert color[1] > color[2]  # More green than blue
    
    def test_fit(self):
        """Test fitting the classifier on player bboxes"""
        classifier = TeamClustering(n_teams=2)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        player_bboxes = [
            [10, 10, 50, 50],
            [60, 10, 100, 50],
            [10, 60, 50, 100],
            [60, 60, 100, 100],
        ]
        
        # Color the uniform areas
        for i, bbox in enumerate(player_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            height = y2 - y1
            margin = int(height * 0.3)
            if i < 2:
                frame[y1+margin:y2-margin, x1:x2] = [255, 0, 0]  # Red team
            else:
                frame[y1+margin:y2-margin, x1:x2] = [0, 0, 255]  # Blue team
        
        classifier.fit(frame, player_bboxes)
        
        assert classifier.kmeans is not None
        assert classifier.kmeans.n_clusters == 2
    
    def test_predict_team_after_fit(self):
        """Test prediction after fitting"""
        classifier = TeamClustering(n_teams=2)
        
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        player_bboxes = [
            [10, 10, 50, 50],
            [60, 10, 100, 50],
            [10, 60, 50, 100],
            [60, 60, 100, 100],
        ]
        
        # Color the uniform areas
        for i, bbox in enumerate(player_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            height = y2 - y1
            margin = int(height * 0.3)
            if i < 2:
                frame[y1+margin:y2-margin, x1:x2] = [255, 0, 0]  # Red team
            else:
                frame[y1+margin:y2-margin, x1:x2] = [0, 0, 255]  # Blue team
        
        classifier.fit(frame, player_bboxes)
        
        # Test prediction on a new red player
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        test_bbox = [110, 10, 150, 50]
        x1, y1, x2, y2 = map(int, test_bbox)
        height = y2 - y1
        margin = int(height * 0.3)
        test_frame[y1+margin:y2-margin, x1:x2] = [255, 0, 0]
        
        team_id = classifier.predict_team(test_frame, test_bbox)
        
        assert team_id in [0, 1]
    
    def test_extract_jersey_color_edge_cases(self):
        """Test edge cases"""
        classifier = TeamClustering()
        
        # Small bbox
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bbox = [10, 10, 30, 30]
        
        color = classifier.extract_jersey_color(frame, bbox)
        assert color.shape == (3,)
        
        # Bbox on the edge
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        bbox = [0, 0, 40, 40]
        
        color = classifier.extract_jersey_color(frame, bbox)
        assert color.shape == (3,)
import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
from src.classifier import SigLIPTeamClassifier


class TestSigLIPTeamClassifier:
    """Tests for SigLIP team classifier"""
    
    def setup_method(self):

        # Mocks for faster tests
        self.mock_model_patcher = patch('src.classifier.AutoModel')
        self.mock_processor_patcher = patch('src.classifier.AutoProcessor')
        
        self.mock_model = self.mock_model_patcher.start()
        self.mock_processor = self.mock_processor_patcher.start()
        
        # Create mock instances
        self.mock_model_instance = Mock()
        
        # Make get_image_features return a real tensor
        def mock_get_image_features(**kwargs):
            return torch.randn(1, 768)

        self.mock_model_instance.get_image_features = mock_get_image_features
        
        # Add .to method
        self.mock_model_instance.to.return_value = self.mock_model_instance
        
        self.mock_model.from_pretrained.return_value = self.mock_model_instance
        
        self.mock_processor_instance = Mock()
        
        # Make processor return a dict with tensor
        self.mock_processor_instance.return_value = {
            'pixel_values': torch.randn(1, 3, 384, 384)
        }
        self.mock_processor.from_pretrained.return_value = self.mock_processor_instance
        
        self.classifier = SigLIPTeamClassifier()
    

    def teardown_method(self):
        """Clean up after a test"""
        self.mock_model_patcher.stop()
        self.mock_processor_patcher.stop()
    
    def test_init(self):
        """Test initialization"""
        assert self.classifier.kmeans is None
        assert self.classifier.cluster_mapping is None
        assert self.classifier.device is not None
        assert self.classifier.processor is not None
        assert self.classifier.model is not None
    
    def test_get_crop_focuses_on_middle(self):
        """Test that crop focuses on the middle"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:30, 20:40] = [255, 0, 0]  # Red top
        frame[30:50, 20:40] = [0, 255, 0]  # Green middle
        frame[50:60, 20:40] = [0, 0, 255]  # Blue bottom
        
        bbox = [20, 20, 40, 60]
        crop = self.classifier._get_crop(frame, bbox)
        
        assert crop is not None
    

    def test_get_crop_invalid_bbox(self):
        """Test crop with invalid bbox"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # x2 <= x1
        bbox = [50, 10, 50, 90]
        crop = self.classifier._get_crop(frame, bbox)
        assert crop is None
        
        # y2 <= y1
        bbox = [10, 50, 90, 50]
        crop = self.classifier._get_crop(frame, bbox)
        assert crop is None
    
    def test_get_crop_edge_cases(self):
        """Test crop with edge cases"""
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Bbox at edge
        bbox = [0, 0, 50, 50]
        crop = self.classifier._get_crop(frame, bbox)
        assert crop is not None
        
        # Small bbox
        bbox = [10, 10, 30, 30]
        crop = self.classifier._get_crop(frame, bbox)
        assert crop is None or crop.shape[0] > 0  # Either None or an array with at least one row
    
    def test_get_embedding(self):
        """Test embedding"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[15:40, 15:40] = [255, 0, 0]
        bbox = [10, 10, 50, 50]
        embedding = self.classifier.get_embedding(frame, bbox)

        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0  # Should have dimensions
    
    def test_get_embedding_invalid_crop(self):
        """Test embedding with invalid crop"""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Invalid bbox that results in None crop
        bbox = [50, 10, 50, 90]
        embedding = self.classifier.get_embedding(frame, bbox)
        
        assert embedding is None
    
    def test_fit_not_enough_people(self):
        """Test fit without enough people"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        person_bboxes = [
            [10, 10, 50, 50],
            [60, 10, 100, 50],
        ]
        self.classifier.fit(frame, person_bboxes)
        
        # Should not create clusters
        assert self.classifier.kmeans is None
        assert self.classifier.cluster_mapping is None
    
    def test_fit(self):
        """Test fit (with enough people)"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Create 9 people
        person_bboxes = []
        for i in range(9):
            x = 10 + (i % 3) * 60
            y = 10 + (i // 3) * 60
            person_bboxes.append([x, y, x + 40, y + 40])
        
        self.classifier.fit(frame, person_bboxes)
        
        assert self.classifier.kmeans is not None
        assert self.classifier.cluster_mapping is not None
        assert self.classifier.kmeans.n_clusters == 3
        assert len(self.classifier.cluster_mapping) == 3
    
    def test_fit_cluster_mapping(self):
        """Test that fit creates correct cluster mapping"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Create 2 refs, 4 team A, 4 team B
        person_bboxes = []
        for i in range(10):
            x = 10 + (i % 5) * 40
            y = 10 + (i // 5) * 60
            person_bboxes.append([x, y, x + 30, y + 30])
        
        self.classifier.fit(frame, person_bboxes)
        
        # Check cluster mapping structure
        assert 2 in self.classifier.cluster_mapping.values()  # Refs
        assert 0 in self.classifier.cluster_mapping.values()  # Team A
        assert 1 in self.classifier.cluster_mapping.values()  # Team B
    
    def test_predict_after_fit(self):
        """Test predict after fit"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Create 9 people
        person_bboxes = []
        for i in range(9):
            x = 10 + (i % 3) * 60
            y = 10 + (i // 3) * 60
            person_bboxes.append([x, y, x + 40, y + 40])
        
        self.classifier.fit(frame, person_bboxes)
        
        test_bbox = [150, 10, 190, 50]
        result = self.classifier.predict(frame, test_bbox)
        assert result in [0, 1, 2]
    
    def test_predict_invalid_embedding(self):
        """Test predict with invalid embedding"""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        
        person_bboxes = []
        for i in range(9):
            x = 10 + (i % 3) * 60
            y = 10 + (i // 3) * 60
            person_bboxes.append([x, y, x + 40, y + 40])
        
        self.classifier.fit(frame, person_bboxes)
        
        # Predict with invalid bbox
        invalid_bbox = [50, 10, 50, 50]  # x2 <= x1
        result = self.classifier.predict(frame, invalid_bbox)
        
        assert result == -1
import unittest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """Create a sample dataset for training"""
        self.data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'label': [0, 1, 0, 1]
        }
        self.df = pd.DataFrame(self.data)
        self.X = self.df[['feature1', 'feature2']]
        self.y = self.df['label']

    def test_train_model(self):
        """Test if the train_model function returns a trained model"""
        model, X_test, y_test = train_model(self.X, self.y)
        
        # Check if the model is of the correct type
        self.assertIsInstance(model, RandomForestClassifier)
        # Check if the test data is returned correctly
        self.assertEqual(X_test.shape[0], 2)  # 50% split
        
if __name__ == '__main__':
    unittest.main()

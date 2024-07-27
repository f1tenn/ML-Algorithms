import unittest
import numpy as np
from src.model_evaluation import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        """Create a sample dataset and train a model"""
        self.X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        self.y = np.array([0, 0, 1, 1])
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(self.X, self.y)

    def test_evaluate_model(self):
        """Test if the evaluate_model function runs without errors"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        try:
            evaluate_model(self.model, X_test, y_test)
        except Exception as e:
            self.fail(f"evaluate_model raised Exception: {e}")

if __name__ == '__main__':
    unittest.main()

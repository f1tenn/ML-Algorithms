import unittest
import pandas as pd
from src.feature_engineering import create_features

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Create a sample dataframe for testing"""
        self.data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        }
        self.df = pd.DataFrame(self.data)

    def test_create_features(self):
        """Test if new features are created correctly"""
        updated_df = create_features(self.df)
        
        # Check if the new feature is created
        self.assertIn('new_feature', updated_df.columns)
        # Check the values of the new feature
        expected_new_feature_values = [4, 10, 18]  # 1*4, 2*5, 3*6
        pd.testing.assert_series_equal(updated_df['new_feature'], pd.Series(expected_new_feature_values))

if __name__ == '__main__':
    unittest.main()

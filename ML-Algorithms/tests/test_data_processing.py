import unittest
import pandas as pd
from src.data_processing import load_data, clean_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        """Create a sample dataframe for testing"""
        self.data = {
            'feature1': [1, 2, None, 4],
            'feature2': [None, 3, 4, 5]
        }
        self.df = pd.DataFrame(self.data)

    def test_load_data(self):
        """Test if the load_data function works correctly"""
        # Save test
        self.df.to_csv('test_data.csv', index=False)
        
        # Load
        loaded_df = load_data('test_data.csv')
        
        # Check if the loaded data
        pd.testing.assert_frame_equal(loaded_df, self.df)

    def test_clean_data(self):
        """Test if the clean_data function correctly removes NaN values"""
        cleaned_df = clean_data(self.df)
        
        # Check if the cleaned dataframe has no NaN values
        self.assertFalse(cleaned_df.isnull().values.any())
        # Check if the number of rows is correct
        self.assertEqual(cleaned_df.shape[0], 2)

if __name__ == '__main__':
    unittest.main()

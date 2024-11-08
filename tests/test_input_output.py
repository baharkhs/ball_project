# test_input_output.py
# Unit tests for input_output.py

# tests/test_input_output.py

import unittest
import os
import json
from ball_simulation.input_output import load_parameters, save_results

class TestInputOutput(unittest.TestCase):
    def setUp(self):
        # Create a temporary JSON file for testing
        self.test_file = 'test_parameters.json'
        with open(self.test_file, 'w') as file:
            json.dump({'mass': 5, 'gravity': 9.81}, file)

    def tearDown(self):
        # Remove the test file after tests
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_parameters(self):
        """Test loading parameters from a JSON file."""
        parameters = load_parameters(self.test_file)
        self.assertEqual(parameters['mass'], 5)
        self.assertEqual(parameters['gravity'], 9.81)

    def test_save_results(self):
        """Test saving results to a JSON file."""
        results = {'time': [0, 1, 2], 'position': [0, 5, 10]}
        result_file = 'test_results.json'
        save_results(results, result_file)

        # Verify that the file was created and contains the correct data
        with open(result_file, 'r') as file:
            saved_results = json.load(file)

        self.assertEqual(saved_results['time'], [0, 1, 2])
        self.assertEqual(saved_results['position'], [0, 5, 10])

        # Clean up
        os.remove(result_file)

if __name__ == '__main__':
    unittest.main()

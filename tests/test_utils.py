import unittest
from ball_simulation.utils import calculate_distance, generate_random_mass

class TestUtils(unittest.TestCase):
    def test_calculate_distance(self):
        dist = calculate_distance((0, 0), (3, 4))
        self.assertEqual(dist, 5)

class TestUtils2(unittest.TestCase):
    def test_generate_random_mass(self):
        mass = generate_random_mass()
        self.assertGreater(mass, 0)  # Ensure mass is positive

if __name__ == '__main__':
    unittest.main()

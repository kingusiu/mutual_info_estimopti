import unittest
from mi_estimopt import example_function

class TestCore(unittest.TestCase):
    def test_example_function(self):
        self.assertEqual(example_function(5), 5)

if __name__ == "__main__":
    unittest.main()

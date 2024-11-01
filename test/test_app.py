import unittest
from src.app import greet


class TestApp(unittest.TestCase):

    def test_greet_with_name(self):
        self.assertEqual(greet("Alice"), "Hello, Alice!")

    def test_greet_without_name(self):
        self.assertEqual(greet(), "Hello, World!")


if __name__ == "__main__":
    unittest.main()

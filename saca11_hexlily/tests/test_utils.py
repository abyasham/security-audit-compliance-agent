import unittest
from utils.utils import split_text

class TestSplitText(unittest.TestCase):
    def test_split_text_basic(self):
        text = "This is a simple test. It should be split into two chunks."
        chunks = split_text(text, chunk_size=10, overlap=2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "This is a simple test.")
        self.assertEqual(chunks[1]["text"], "simple test. It should be split into two chunks.")
        self.assertEqual(chunks[0]["source_type"], None)
        self.assertEqual(chunks[0]["source_name"], None)
        self.assertEqual(chunks[1]["source_type"], None)
        self.assertEqual(chunks[1]["source_name"], None)

    def test_split_text_with_source(self):
        text = "This is another test. It includes source information."
        chunks = split_text(text, chunk_size=10, overlap=2, source_type="log", source_name="test_log.txt")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "This is another test.")
        self.assertEqual(chunks[1]["text"], "another test. It includes source information.")
        self.assertEqual(chunks[0]["source_type"], "log")
        self.assertEqual(chunks[0]["source_name"], "test_log.txt")
        self.assertEqual(chunks[1]["source_type"], "log")
        self.assertEqual(chunks[1]["source_name"], "test_log.txt")

    def test_split_text_binary(self):
        binary_text = "A" * 20  # Binary-like text
        chunks = split_text(binary_text, chunk_size=10, overlap=2)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "A" * 10)
        self.assertEqual(chunks[1]["text"], "A" * 10)
        self.assertEqual(chunks[0]["source_type"], None)
        self.assertEqual(chunks[0]["source_name"], None)
        self.assertEqual(chunks[1]["source_type"], None)
        self.assertEqual(chunks[1]["source_name"], None)

if __name__ == '__main__':
    unittest.main()
import unittest
from utils.utils import split_text

class TestSplitText(unittest.TestCase):
    def test_text_splitting(self):
        text = "This is a test. This is only a test. In the event of an actual emergency..."
        chunks = split_text(text, chunk_size=10, overlap=2, source_type="log", source_name="test_log.txt")
        expected_chunks = [
            {"text": "This is a test. This is", "source_type": "log", "source_name": "test_log.txt"},
            {"text": "This is only a test. In", "source_type": "log", "source_name": "test_log.txt"},
            {"text": "In the event of an actual", "source_type": "log", "source_name": "test_log.txt"}
        ]
        self.assertEqual(chunks, expected_chunks)

    def test_binary_splitting(self):
        binary_text = "0101010101010101010101010101010101010101010101010101010101010101"
        chunks = split_text(binary_text, chunk_size=10, overlap=2, source_type="binary", source_name="test_binary.bin")
        expected_chunks = [
            {"text": "0101010101", "source_type": "binary", "source_name": "test_binary.bin"},
            {"text": "010101010101010101", "source_type": "binary", "source_name": "test_binary.bin"},
            {"text": "010101010101010101010101", "source_type": "binary", "source_name": "test_binary.bin"},
            {"text": "010101010101010101010101", "source_type": "binary", "source_name": "test_binary.bin"}
        ]
        self.assertEqual(chunks, expected_chunks)

if __name__ == '__main__':
    unittest.main()
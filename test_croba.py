import unittest
import torch
from croba import ByteTokenizer, Croba, CrobaConfig


class CrobaTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        text = "Croba says hello 🦀"
        self.assertEqual(ByteTokenizer.decode(ByteTokenizer.encode(text)), text)

    def test_model_shape_and_loss(self):
        cfg = CrobaConfig(context_length=8, n_layers=1, n_heads=2, d_model=16, dropout=0)
        model = Croba(cfg)
        x = torch.randint(0, 256, (2, 8))
        logits, loss = model(x, x)
        self.assertEqual(logits.shape, (2, 8, 256))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

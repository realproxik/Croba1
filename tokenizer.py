"""A lossless UTF-8 byte tokenizer with optional Rust acceleration."""

try:
    import croba_tokenizer_rs as _rust
except ImportError:
    _rust = None


class ByteTokenizer:
    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        if _rust is not None:
            return _rust.encode(text)
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        if _rust is not None:
            return _rust.decode(tokens)
        return bytes(tokens).decode("utf-8", errors="replace")


import pytest

from app.domain import MemoryClue


def test_memory_clue_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        MemoryClue(raw_text="鞋", normalized_text="鞋", confidence=1.1)


import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from train_tokenizer import train_tokenizer
from transformers import AutoTokenizer


def test_bilingual_tokenizer_smoke(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    bilingual = (
        "Mathematics and literature explain patterns in human experience.\n" * 100
        + "数学、物理与文学共同描述人类经验中的规律。\n" * 100
    )
    (books / "bilingual.txt").write_text(bilingual, encoding="utf-8")
    manifest = {
        "tokenizer_vocab_size": 320,
        "tokenizer_max_documents_per_source": 10,
        "tokenizer_max_chars_per_source": 100_000,
        "min_chars": 10,
        "sources": [
            {
                "name": "books",
                "path": str(books / "*.txt"),
                "format": "text",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "tokenizer"
    train_tokenizer(manifest_path, output)
    tokenizer = AutoTokenizer.from_pretrained(output)
    assert len(tokenizer) == 320
    text = "Physics 与小说"
    assert tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True) == text

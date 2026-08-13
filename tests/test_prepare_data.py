import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from prepare_data import (
    collapse_repeated_document,
    iter_local_documents,
    manifest_fingerprint,
    record_matches_filters,
    strip_gutenberg_boilerplate,
)


def test_plain_text_file_is_one_group(tmp_path):
    book = tmp_path / "novel.txt"
    book.write_text("Chapter I\n\nA sufficiently long book body.", encoding="utf-8")
    documents = list(
        iter_local_documents(
            {"name": "books", "path": str(tmp_path / "*.txt"), "format": "text"}
        )
    )
    assert len(documents) == 1
    assert documents[0].group_id == "novel.txt"
    assert documents[0].text.startswith("Chapter I")


def test_gutenberg_wrapper_is_removed():
    wrapped = (
        "licensing notes\n*** START OF THE PROJECT GUTENBERG EBOOK NOVEL ***\n"
        "Chapter I\nBody\n*** END OF THE PROJECT GUTENBERG EBOOK NOVEL ***\n"
        "more licensing notes"
    )
    assert strip_gutenberg_boilerplate(wrapped).strip() == "Chapter I\nBody"


def test_concatenated_complete_copies_are_collapsed():
    body = "BOOK TITLE\n\n" + "A unique sentence in the story.\n" * 3_000
    collapsed, removed = collapse_repeated_document(body + "\n\n" + body)
    assert removed == 1
    assert collapsed == body.rstrip()


def test_manifest_numeric_filters():
    source = {
        "filters": [
            {"field": "score", "op": "gte", "value": 0.8},
            {"field": "metadata.cluster", "op": "lte", "value": 10},
        ]
    }
    assert record_matches_filters(
        {"score": 0.91, "metadata": {"cluster": 3}}, source
    )
    assert not record_matches_filters(
        {"score": 0.79, "metadata": {"cluster": 3}}, source
    )


def test_manifest_fingerprint_is_stable_and_content_sensitive():
    first = {"seed": 42, "sources": [{"name": "books", "max_tokens": 100}]}
    reordered = {
        "sources": [{"max_tokens": 100, "name": "books"}],
        "seed": 42,
    }
    changed = {"seed": 42, "sources": [{"name": "books", "max_tokens": 101}]}
    assert manifest_fingerprint(first) == manifest_fingerprint(reordered)
    assert manifest_fingerprint(first) != manifest_fingerprint(changed)

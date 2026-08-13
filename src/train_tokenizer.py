"""Train the project's bilingual byte-level BPE tokenizer from manifest sources."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from prepare_data import (
    collapse_repeated_document,
    iter_source_documents,
    load_manifest,
    normalize_text,
    strip_gutenberg_boilerplate,
)


EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"


def tokenizer_corpus(manifest: dict) -> Iterator[str]:
    seed = int(manifest.get("seed", 42))
    default_documents = int(manifest.get("tokenizer_max_documents_per_source", 20_000))
    default_chars = int(manifest.get("tokenizer_max_chars_per_source", 20_000_000))
    max_chars_per_document = int(
        manifest.get("tokenizer_max_chars_per_document", 100_000)
    )
    for source_index, source in enumerate(manifest["sources"]):
        if source.get("use_for_tokenizer", True) is False:
            continue
        document_limit = int(source.get("tokenizer_max_documents", default_documents))
        char_limit = int(source.get("tokenizer_max_chars", default_chars))
        source_chars = 0
        source_documents = 0
        print(
            f"tokenizer source={source['name']} max_documents={document_limit:,} "
            f"max_chars={char_limit:,}"
        )
        for document in iter_source_documents(source, seed + source_index):
            text = document.text
            if source.get("strip_gutenberg_boilerplate", False):
                text = strip_gutenberg_boilerplate(text)
            text = normalize_text(text)
            if source.get("collapse_repeated_documents", False):
                text, _ = collapse_repeated_document(text)
            if len(text) < int(manifest.get("min_chars", 200)):
                continue
            remaining = char_limit - source_chars
            if remaining <= 0 or source_documents >= document_limit:
                break
            sample = text[: min(max_chars_per_document, remaining)]
            if sample:
                yield sample
                source_chars += len(sample)
                source_documents += 1
        print(
            f"tokenizer source={source['name']} documents={source_documents:,} "
            f"chars={source_chars:,}"
        )


def train_tokenizer(manifest_path: str | Path, output_dir: str | Path) -> None:
    manifest = load_manifest(manifest_path)
    vocab_size = int(manifest.get("tokenizer_vocab_size", 50_304))
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=[EOS_TOKEN, UNK_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(tokenizer_corpus(manifest), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=EOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
    )
    if len(fast) != vocab_size:
        raise ValueError(f"trained vocab has {len(fast)} entries, expected {vocab_size}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(output)
    print(f"saved bilingual tokenizer vocab_size={len(fast):,} to {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/data_sources.json")
    parser.add_argument("--output-dir", default="tokenizers/minigpt-bilingual-50304")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train_tokenizer(arguments.manifest, arguments.output_dir)

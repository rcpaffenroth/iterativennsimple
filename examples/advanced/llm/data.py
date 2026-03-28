"""WikiText dataset download, tokenization, and DataLoader creation.

Supports WikiText-2 (quick experiments) and WikiText-103 (large-scale training).
Uses tiktoken with GPT-2 encoding (50257 vocab).

Usage::

    from data import load_wikitext, create_dataloaders

    train_tokens, val_tokens, test_tokens = load_wikitext("2")
    train_loader, val_loader = create_dataloaders(
        train_tokens, val_tokens, context_len=256, batch_size=32,
    )
"""

import os
import ssl
import urllib.request
import zipfile

import numpy as np
import torch

# WikiText download URLs (raw text, zipped)
# HuggingFace community mirrors — the original S3 URLs are no longer available.
_WIKITEXT_URLS = {
    "2": "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip",
    "103": "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip",
}

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "iterativennsimple")


def _download_and_extract(version: str) -> str:
    """Download and extract WikiText, returning the extracted directory path."""
    assert version in _WIKITEXT_URLS, f"Unknown version: {version}. Use '2' or '103'."

    cache_dir = os.path.join(_CACHE_DIR, f"wikitext-{version}-raw")
    # Check if already extracted
    subdir = f"wikitext-{version}-raw"
    extracted = os.path.join(_CACHE_DIR, subdir)
    if os.path.isdir(extracted):
        return extracted

    os.makedirs(_CACHE_DIR, exist_ok=True)
    url = _WIKITEXT_URLS[version]
    zip_path = os.path.join(_CACHE_DIR, f"wikitext-{version}-raw-v1.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading WikiText-{version} from {url} ...")
        # HuggingFace CDN may redirect; use a custom opener that follows redirects
        ctx = ssl.create_default_context()
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx),
            urllib.request.HTTPRedirectHandler(),
        )
        req = urllib.request.Request(url, headers={"User-Agent": "iterativennsimple/0.4"})
        with opener.open(req) as resp, open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB chunks
                if not chunk:
                    break
                f.write(chunk)
        print(f"  saved to {zip_path}")

    print(f"Extracting to {_CACHE_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(_CACHE_DIR)

    assert os.path.isdir(extracted), f"Expected directory {extracted} after extraction."
    return extracted


def _read_split(data_dir: str, split: str) -> str:
    """Read a single split (train/valid/test) from the extracted directory."""
    path = os.path.join(data_dir, f"wiki.{split}.raw")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _tokenize(text: str) -> np.ndarray:
    """Tokenize text using tiktoken GPT-2 encoding."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    return np.array(tokens, dtype=np.int64)


def load_wikitext(version: str = "2", verbose: bool = True):
    """Download (if needed) and tokenize WikiText.

    Args:
        version: "2" for WikiText-2, "103" for WikiText-103.
        verbose: Print download/tokenization progress.

    Returns:
        (train_tokens, val_tokens, test_tokens) as numpy int64 arrays.
    """
    data_dir = _download_and_extract(version)

    # Check for cached token files
    cache_prefix = os.path.join(_CACHE_DIR, f"wikitext-{version}-tokens")
    splits = ["train", "valid", "test"]
    cached = {s: f"{cache_prefix}.{s}.npy" for s in splits}

    if all(os.path.exists(cached[s]) for s in splits):
        if verbose:
            print(f"Loading cached tokens from {cache_prefix}.*.npy ...")
        return tuple(np.load(cached[s]) for s in splits)

    if verbose:
        print("Tokenizing with tiktoken GPT-2 encoding ...")

    results = []
    for split in splits:
        text = _read_split(data_dir, split)
        tokens = _tokenize(text)
        np.save(cached[split], tokens)
        if verbose:
            print(f"  {split}: {len(tokens):,} tokens")
        results.append(tokens)

    return tuple(results)


class TokenDataset(torch.utils.data.Dataset):
    """Dataset of fixed-length token windows for language modeling.

    Each item is (input_ids, target_ids) where target = input shifted by 1.
    """

    def __init__(self, tokens: np.ndarray, context_len: int):
        self.tokens = torch.from_numpy(tokens).long()
        self.context_len = context_len
        # Number of complete windows (non-overlapping for training)
        self.n_windows = len(self.tokens) // (context_len + 1)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * (self.context_len + 1)
        chunk = self.tokens[start : start + self.context_len + 1]
        return chunk[:-1], chunk[1:]


def create_dataloaders(
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
    context_len: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Create train and validation DataLoaders from token arrays.

    Args:
        train_tokens: 1D numpy array of training token ids.
        val_tokens: 1D numpy array of validation token ids.
        context_len: Sequence length for each window.
        batch_size: Batch size.
        num_workers: DataLoader workers.

    Returns:
        (train_loader, val_loader, vocab_size)
    """
    train_ds = TokenDataset(train_tokens, context_len)
    val_ds = TokenDataset(val_tokens, context_len)

    vocab_size = 50257  # GPT-2 vocab size

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, vocab_size

"""Download text data, preprocess it, and train a tiny model from random weights.

No pretrained model or tokenizer is downloaded. The default pipeline first
trains a language model on WikiText-103, then continues training it on OASST1
conversation paths. Use --prepare-only to download and prepare data without
starting model training.
"""

from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from slm import TinyDecoder, generate


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
WIKI_ZIP = RAW_DIR / "wikitext-103-raw-v1.zip"
WIKI_DIR = PROCESSED_DIR / "wikitext-103"
OASST_GZIP = RAW_DIR / "oasst1_ready.messages.jsonl.gz"
ARTIFACTS = ROOT / "artifacts" / "scratch_chat"

WIKI_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
WIKI_SHA256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794"
OASST_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    "f8b2b76b6c50a1e01b8c2d0cf869513cb04e78b1/"
    "2023-04-12_oasst_ready.messages.jsonl.gz?download=true"
)

SEED = 17
MAX_WIKI_DOWNLOAD_BYTES = 250_000_000
MAX_WIKI_EXTRACTED_BYTES = 600_000_000
MAX_OASST_DOWNLOAD_BYTES = 60_000_000
MAX_MESSAGE_CHARACTERS = 3_000
MAX_CONVERSATION_MESSAGES = 8

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endoftext|>"

MODEL_CONFIG = {
    "block_size": 128,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 192,
    "dropout": 0.1,
}
VOCAB_SIZE = 4_096
BATCH_SIZE = 2
PRETRAIN_STEPS = 10_000
CHAT_STEPS = 3_000
EVAL_INTERVAL = 250
EVAL_BATCHES = 8
PRETRAIN_LEARNING_RATE = 3e-4
CHAT_LEARNING_RATE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, max_bytes: int, expected_sha256=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        size_ok = 100_000 <= destination.stat().st_size <= max_bytes
        hash_ok = expected_sha256 is None or sha256_file(destination) == expected_sha256
        if size_ok and hash_ok:
            print(f"Using existing dataset file: {destination}")
            return
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    total = 0
    print(f"Downloading text dataset: {destination.name}")
    request = urllib.request.Request(url, headers={"User-Agent": "scratch-slm-research-project"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1_048_576)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Download exceeded the {max_bytes:,}-byte safety limit.")
                output.write(chunk)
                print(f"  {total / 1_000_000:.1f} MB", end="\r", flush=True)
        if total < 100_000:
            raise ValueError(f"Downloaded file is unexpectedly small ({total:,} bytes).")
        if expected_sha256 and sha256_file(partial) != expected_sha256:
            raise ValueError("Downloaded WikiText archive failed its SHA-256 check.")
        partial.replace(destination)
        print(f"Downloaded {total / 1_000_000:.1f} MB to {destination}")
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Check your internet connection and try again."
        ) from exc


def ensure_wikitext():
    split_paths = {name: WIKI_DIR / f"{name}.txt" for name in ("train", "valid", "test")}
    if all(path.is_file() and path.stat().st_size > 100_000 for path in split_paths.values()):
        print("Using existing preprocessed WikiText files.")
        return split_paths

    download_file(WIKI_URL, WIKI_ZIP, MAX_WIKI_DOWNLOAD_BYTES, WIKI_SHA256)
    expected_names = {
        "train": "wiki.train.tokens",
        "valid": "wiki.valid.tokens",
        "test": "wiki.test.tokens",
    }
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(WIKI_ZIP) as archive:
            matches = {}
            for info in archive.infolist():
                basename = Path(info.filename.replace("\\", "/")).name
                for split, expected_name in expected_names.items():
                    if basename == expected_name:
                        if split in matches:
                            raise ValueError(f"Archive contains duplicate {expected_name} files.")
                        matches[split] = info
            if set(matches) != set(expected_names):
                raise ValueError("Archive does not contain all three expected WikiText splits.")

            for split, info in matches.items():
                total_uncompressed += info.file_size
                if info.file_size <= 0 or total_uncompressed > MAX_WIKI_EXTRACTED_BYTES:
                    raise ValueError("WikiText archive has an unexpected extracted size.")
                destination = split_paths[split]
                temporary = destination.with_suffix(destination.suffix + ".part")
                with archive.open(info) as source, temporary.open("wb") as output:
                    copied = 0
                    while True:
                        chunk = source.read(1_048_576)
                        if not chunk:
                            break
                        copied += len(chunk)
                        if copied > info.file_size or copied > MAX_WIKI_EXTRACTED_BYTES:
                            raise ValueError("Extracted WikiText file exceeded its expected size.")
                        output.write(chunk)
                if copied != info.file_size:
                    raise ValueError(f"Incomplete extraction of {info.filename}.")
                temporary.replace(destination)
                print(f"Prepared WikiText {split}: {destination.stat().st_size / 1_000_000:.1f} MB")
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        for path in split_paths.values():
            path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
        raise RuntimeError(f"Could not safely extract WikiText: {exc}") from exc
    return split_paths


def iter_wiki_text(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.strip()
            if text:
                yield text


def load_oasst_conversations():
    download_file(OASST_URL, OASST_GZIP, MAX_OASST_DOWNLOAD_BYTES)
    records = {}
    uncompressed_bytes = 0
    with gzip.open(OASST_GZIP, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            uncompressed_bytes += len(line.encode("utf-8"))
            if uncompressed_bytes > 1_000_000_000:
                raise ValueError("Expanded OASST1 data exceeded the 1 GB safety limit.")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in OASST1 at line {line_number}.") from exc
            text = item.get("text")
            if (
                item.get("lang") != "en"
                or item.get("deleted")
                or item.get("synthetic") is True
                or item.get("review_result") is False
                or item.get("role") not in {"prompter", "assistant"}
                or not isinstance(text, str)
                or not text.strip()
                or len(text) > MAX_MESSAGE_CHARACTERS
            ):
                continue
            records[item["message_id"]] = item

    children = defaultdict(list)
    roots = []
    for item in records.values():
        parent_id = item.get("parent_id")
        if parent_id in records:
            children[parent_id].append(item)
        elif not parent_id and item.get("role") == "prompter":
            roots.append(item)

    conversations = []
    seen_paths = set()
    stack = [(root, []) for root in roots]
    while stack:
        item, parent_path = stack.pop()
        if any(previous["message_id"] == item["message_id"] for previous in parent_path):
            continue
        path = parent_path + [item]
        expected_role = "assistant" if item["role"] == "prompter" else "prompter"
        options = [child for child in children.get(item["message_id"], [])
                   if child.get("role") == expected_role]
        if expected_role == "assistant":
            options = [child for child in options if child.get("rank") in (None, 0)]
        if options and len(path) < 100:
            stack.extend((child, path) for child in options)
            continue
        if len(path) < 2 or path[-1].get("role") != "assistant":
            continue

        trimmed = path[-MAX_CONVERSATION_MESSAGES:]
        if trimmed[0].get("role") == "assistant":
            trimmed = trimmed[1:]
        tree_id = trimmed[-1].get("message_tree_id")
        if not tree_id:
            continue
        turns = []
        for message in trimmed:
            role_token = USER_TOKEN if message["role"] == "prompter" else ASSISTANT_TOKEN
            turns.append(role_token + message["text"].strip())
        document = "\n".join(turns)
        identity = (tree_id, document)
        if identity not in seen_paths:
            seen_paths.add(identity)
            conversations.append((tree_id, document))

    if len(conversations) < 500:
        raise ValueError(
            f"Only {len(conversations):,} usable English conversations were prepared. "
            "Check the OASST1 archive and filtering rules."
        )

    tree_ids = sorted({tree_id for tree_id, _ in conversations})
    random.Random(SEED).shuffle(tree_ids)
    validation_count = max(1, int(len(tree_ids) * 0.05))
    validation_ids = set(tree_ids[:validation_count])
    train_docs = [doc for tree_id, doc in conversations if tree_id not in validation_ids]
    valid_docs = [doc for tree_id, doc in conversations if tree_id in validation_ids]
    if not train_docs or not valid_docs:
        raise ValueError("OASST1 conversation train/validation split is empty.")
    print(
        f"OASST1 preprocessing: {len(records):,} filtered messages, "
        f"{len(conversations):,} conversation paths, {len(train_docs):,} train, "
        f"{len(valid_docs):,} validation."
    )
    return train_docs, valid_docs


def make_tokenizer(wiki_train: Path, chat_train_docs: list[str], destination: Path,
                   source_signature: str):
    metadata_path = destination.with_name("tokenizer_sources.json")
    if destination.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_signature") == source_signature:
            print(f"Using existing tokenizer: {destination}")
            return Tokenizer.from_file(str(destination))

    if destination.is_file():
        print("Training a fresh tokenizer because the prepared data changed.")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<unk>", USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def texts():
        yield from iter_wiki_text(wiki_train)
        yield from chat_train_docs

    print("Training a new BPE tokenizer from WikiText and OASST1 text...")
    tokenizer.train_from_iterator(texts(), trainer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    tokenizer.save(str(temporary))
    temporary.replace(destination)
    atomic_json_save({"source_signature": source_signature}, metadata_path)
    return tokenizer


def encode_wiki(path: Path, tokenizer: Tokenizer, end_id: int):
    values = array("I")
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = line.rstrip("\r\n")
            values.extend(tokenizer.encode(text + "\n").ids)
            if not text.strip():
                values.append(end_id)
    values.append(end_id)
    return torch.frombuffer(values, dtype=torch.int32).to(dtype=torch.long)


def encode_conversations(documents: list[str], tokenizer: Tokenizer, end_id: int):
    values = array("I")
    for document in documents:
        values.extend(tokenizer.encode(document).ids)
        values.append(end_id)
    return torch.frombuffer(values, dtype=torch.int32).to(dtype=torch.long)


def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device):
    if len(tokens) <= block_size + 1:
        raise ValueError(f"Need more than {block_size + 1:,} tokens; found {len(tokens):,}.")
    starts = torch.randint(len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([tokens[int(i):int(i) + block_size] for i in starts]).to(device)
    y = torch.stack([tokens[int(i) + 1:int(i) + block_size + 1] for i in starts]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_tokens, valid_tokens, batch_size, config, device):
    model.eval()
    results = {}
    for name, tokens in (("train", train_tokens), ("validation", valid_tokens)):
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(tokens, batch_size, config["block_size"], device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def atomic_torch_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def train_stage(name, model, train_tokens, valid_tokens, steps, learning_rate,
                batch_size, config, device, data_signature):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    last_path = ARTIFACTS / f"{name}_last.pt"
    best_path = ARTIFACTS / f"{name}_best.pt"
    history_path = ARTIFACTS / f"{name}_loss_history.json"
    signature = hashlib.sha256(json.dumps({
        "name": name,
        "data": data_signature,
        "config": config,
        "steps": steps,
        "lr": learning_rate,
        "batch": batch_size,
        "seed": SEED,
    }, sort_keys=True).encode()).hexdigest()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step, best_validation, history = 0, float("inf"), []
    if last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=True)
        if saved.get("signature") != signature:
            raise RuntimeError(
                f"{last_path.name} belongs to different data/settings. Move "
                f"{ARTIFACTS} aside before starting with changed settings."
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"]) + 1
        best_validation = float(saved["best_validation"])
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resuming {name} at step {start_step:,}.")

    for step in range(start_step, steps):
        model.train()
        x, y = get_batch(train_tokens, batch_size, config["block_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == steps:
            measured = estimate_loss(
                model, train_tokens, valid_tokens, batch_size, config, device
            )
            history.append({"step": step + 1, **measured})
            atomic_json_save(history, history_path)
            if measured["validation"] < best_validation:
                best_validation = measured["validation"]
                atomic_torch_save(model.state_dict(), best_path)
            atomic_torch_save({
                "signature": signature,
                "step": step,
                "best_validation": best_validation,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, last_path)
            perplexity = math.exp(min(measured["validation"], 20))
            print(
                f"{name} {step + 1:>6,}/{steps:,}: "
                f"train loss {measured['train']:.3f}, "
                f"validation loss {measured['validation']:.3f}, "
                f"validation perplexity {perplexity:.2f}"
            )

    if not best_path.is_file():
        raise RuntimeError(f"{name} did not produce a checkpoint.")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Download and preprocess data, then stop before training.")
    parser.add_argument("--pretrain-steps", type=int, default=PRETRAIN_STEPS)
    parser.add_argument("--chat-steps", type=int, default=CHAT_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if min(args.pretrain_steps, args.chat_steps, args.batch_size) < 1:
        parser.error("step counts and batch size must be positive integers")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__}; device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is unavailable; training will use CPU and may be very slow.")

    wiki_paths = ensure_wikitext()
    chat_train_docs, chat_valid_docs = load_oasst_conversations()
    corpus_digest = hashlib.sha256()
    for path in (wiki_paths["train"], wiki_paths["valid"], OASST_GZIP):
        corpus_digest.update(sha256_file(path).encode("ascii"))
    source_signature = corpus_digest.hexdigest()
    tokenizer_path = ARTIFACTS / "tokenizer.json"
    tokenizer = make_tokenizer(
        wiki_paths["train"], chat_train_docs, tokenizer_path, source_signature
    )
    end_id = tokenizer.token_to_id(END_TOKEN)
    user_id = tokenizer.token_to_id(USER_TOKEN)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
    if end_id is None or user_id is None or assistant_id is None:
        raise RuntimeError("The tokenizer is missing one or more required chat tokens.")

    if args.prepare_only:
        print("Data and tokenizer are ready. Training was skipped (--prepare-only).")
        return

    wiki_train = encode_wiki(wiki_paths["train"], tokenizer, end_id)
    wiki_valid = encode_wiki(wiki_paths["valid"], tokenizer, end_id)
    chat_train = encode_conversations(chat_train_docs, tokenizer, end_id)
    chat_valid = encode_conversations(chat_valid_docs, tokenizer, end_id)
    model_config = {
        **MODEL_CONFIG,
        "vocab_size": tokenizer.get_vocab_size(),
        "end_token_id": end_id,
        "user_token_id": user_id,
        "assistant_token_id": assistant_id,
    }
    model = TinyDecoder(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Randomly initialized model parameters: {parameter_count:,}")
    print(f"Model configuration: {model_config}")
    print(f"WikiText tokens: {len(wiki_train):,} train; {len(wiki_valid):,} validation")
    print(f"OASST1 tokens: {len(chat_train):,} train; {len(chat_valid):,} validation")
    print(f"Training with batch size {args.batch_size}; context {MODEL_CONFIG['block_size']}.")

    data_signature = hashlib.sha256(
        (source_signature + sha256_file(tokenizer_path)).encode("ascii")
    ).hexdigest()
    pretrain_history = train_stage(
        "pretrain", model, wiki_train, wiki_valid, args.pretrain_steps,
        PRETRAIN_LEARNING_RATE, args.batch_size, model_config, device, data_signature,
    )
    chat_history = train_stage(
        "chat", model, chat_train, chat_valid, args.chat_steps,
        CHAT_LEARNING_RATE, args.batch_size, model_config, device, data_signature,
    )

    config_to_save = {
        **model_config,
        "parameter_count": parameter_count,
        "from_scratch": True,
        "pretraining_dataset": "WikiText-103 raw v1",
        "chat_dataset": "OpenAssistant OASST1",
        "data_signature": data_signature,
        "limitations": (
            "Small educational model trained on limited public text; not a dependable "
            "general-purpose assistant. Verify important claims independently."
        ),
    }
    model.eval()
    atomic_torch_save(model.state_dict(), ARTIFACTS / "model.pt")
    atomic_json_save(config_to_save, ARTIFACTS / "config.json")
    atomic_json_save({"pretrain": pretrain_history, "chat": chat_history},
                     ARTIFACTS / "loss_history.json")
    print(f"\nExported model to {ARTIFACTS / 'model.pt'}")
    prompt = USER_TOKEN + "Say hello in one short sentence.\n" + ASSISTANT_TOKEN
    sample = generate(model, tokenizer, prompt, end_id, max_new_tokens=48,
                      temperature=0.7, stop_token_ids=[user_id])
    print("\nGenerated sample (not a verified factual answer):")
    print(sample)


if __name__ == "__main__":
    main()

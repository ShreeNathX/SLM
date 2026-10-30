"""Train a small chat language model from random weights and open conversation data.

This script downloads training text only. It never downloads model weights.
Run it from the project folder or through train_chat_from_scratch.ipynb.
"""

from collections import defaultdict
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
import urllib.error
import urllib.request

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from slm import TinyDecoder, generate


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PATH = RAW_DIR / "oasst1_ready.messages.jsonl.gz"
ARTIFACTS = ROOT / "artifacts" / "chat"

# The OpenAssistant OASST1 corpus is Apache-2.0 licensed. This pinned URL is
# the conversation text only; no model checkpoint or tokenizer is downloaded.
SOURCE_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/"
    "f8b2b76b6c50a1e01b8c2d0cf869513cb04e78b1/"
    "2023-04-12_oasst_ready.messages.jsonl.gz?download=true"
)
MAX_DOWNLOAD_BYTES = 60_000_000
MAX_MESSAGE_CHARACTERS = 3_000
MAX_CONVERSATION_MESSAGES = 8

SEED = 17
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, torch.get_num_threads()))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__}; device: {DEVICE}")
if DEVICE.type == "cpu":
    print("CPU training selected. This run may take a while; checkpoints allow it to resume.")

RAW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)
if not RAW_PATH.exists():
    try:
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"User-Agent": "from-scratch-local-chat-model"},
        )
        temporary_raw_path = RAW_PATH.with_suffix(RAW_PATH.suffix + ".part")
        total_bytes = 0
        with urllib.request.urlopen(request, timeout=300) as response:
            with temporary_raw_path.open("wb") as output:
                while True:
                    chunk = response.read(1_048_576)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_DOWNLOAD_BYTES:
                        raise ValueError("The downloaded training corpus exceeded the size limit.")
                    output.write(chunk)
                    print(f"Downloaded training text: {total_bytes / 1_000_000:.1f} MB", flush=True)
        temporary_raw_path.replace(RAW_PATH)
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(
            f"Could not obtain the training corpus. Download {SOURCE_URL} in a browser "
            f"and save it to {RAW_PATH}. This is training data, not a model."
        ) from exc

raw_bytes = RAW_PATH.read_bytes()
if not 100_000 <= len(raw_bytes) <= MAX_DOWNLOAD_BYTES:
    raise ValueError(f"Training corpus size is invalid: {len(raw_bytes):,} bytes.")
corpus_sha256 = hashlib.sha256(raw_bytes).hexdigest()

# Keep English, non-deleted, non-synthetic messages and exclude messages
# explicitly rejected during review. The ready split is already moderated.
records = {}
with gzip.open(RAW_PATH, "rt", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on corpus line {line_number}.") from exc
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


def preferred_children(item):
    options = children.get(item["message_id"], [])
    expected_role = "assistant" if item["role"] == "prompter" else "prompter"
    options = [child for child in options if child.get("role") == expected_role]
    if expected_role == "assistant":
        preferred = [child for child in options if child.get("rank") in (None, 0)]
        return preferred
    return options


conversation_docs = []
seen_paths = set()


def collect_leaf_paths(item, path):
    path = path + [item]
    next_messages = preferred_children(item)
    if not next_messages:
        if len(path) >= 2 and path[-1].get("role") == "assistant":
            trimmed = path[-MAX_CONVERSATION_MESSAGES:]
            if trimmed[0].get("role") == "assistant":
                trimmed = trimmed[1:]
            tree_id = trimmed[-1].get("message_tree_id")
            if tree_id:
                turns = []
                for message in trimmed:
                    role_token = "<|user|>" if message["role"] == "prompter" else "<|assistant|>"
                    turns.append(role_token + message["text"].strip())
                document = "\n".join(turns)
                identity = (tree_id, document)
                if identity not in seen_paths:
                    seen_paths.add(identity)
                    conversation_docs.append((tree_id, document))
        return
    for child in next_messages:
        collect_leaf_paths(child, path)


for root in roots:
    collect_leaf_paths(root, [])

if len(conversation_docs) < 500:
    raise ValueError(
        f"Only {len(conversation_docs):,} usable English conversations were built. "
        "Check the corpus download and filtering rules."
    )

tree_ids = sorted({tree_id for tree_id, _ in conversation_docs})
random.Random(SEED).shuffle(tree_ids)
validation_tree_count = max(1, int(0.05 * len(tree_ids)))
validation_tree_ids = set(tree_ids[:validation_tree_count])
train_docs = [text for tree_id, text in conversation_docs if tree_id not in validation_tree_ids]
val_docs = [text for tree_id, text in conversation_docs if tree_id in validation_tree_ids]
if not train_docs or not val_docs:
    raise ValueError("Conversation training and validation sets must both be non-empty.")

print(f"Compressed corpus bytes: {len(raw_bytes):,}; SHA-256: {corpus_sha256}")
print(f"Filtered messages: {len(records):,}; conversation paths: {len(conversation_docs):,}")
print(f"Training paths: {len(train_docs):,}; validation paths: {len(val_docs):,}")

ARTIFACTS.mkdir(parents=True, exist_ok=True)
TOKENIZER_PATH = ARTIFACTS / "tokenizer.json"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|endoftext|>"

tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
token_trainer = trainers.BpeTrainer(
    vocab_size=2_048,
    min_frequency=2,
    special_tokens=["<unk>", USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train_from_iterator(train_docs, token_trainer)
end_token_id = tokenizer.token_to_id(END_TOKEN)
user_token_id = tokenizer.token_to_id(USER_TOKEN)
assistant_token_id = tokenizer.token_to_id(ASSISTANT_TOKEN)
if end_token_id is None or user_token_id is None or assistant_token_id is None:
    raise RuntimeError("The tokenizer did not create the required chat tokens.")


def encode_documents(documents):
    pieces = []
    for document in documents:
        pieces.extend(tokenizer.encode(document).ids)
        pieces.append(end_token_id)
    return torch.tensor(pieces, dtype=torch.long)


train_tokens = encode_documents(train_docs)
val_tokens = encode_documents(val_docs)

MODEL_CONFIG = {
    "vocab_size": tokenizer.get_vocab_size(),
    "block_size": 256,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 128,
    "dropout": 0.1,
}
model = TinyDecoder(MODEL_CONFIG).to(DEVICE)
parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"Randomly initialized parameters: {parameter_count:,}")
print(f"Configuration: {MODEL_CONFIG}")
print(f"Training tokens: {len(train_tokens):,}; validation tokens: {len(val_tokens):,}")

MAX_STEPS = 6_000
EVAL_INTERVAL = 300
EVAL_BATCHES = 8
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
RESUME = True


def get_batch(tokens, batch_size, block_size):
    if len(tokens) <= block_size:
        raise ValueError(f"Need more than {block_size} tokens, found {len(tokens)}.")
    starts = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i:i + block_size] for i in starts]).to(DEVICE)
    y = torch.stack([tokens[i + 1:i + block_size + 1] for i in starts]).to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(current_model, train_stream, val_stream):
    current_model.eval()
    values = {}
    for name, stream in (("train", train_stream), ("validation", val_stream)):
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(stream, BATCH_SIZE, MODEL_CONFIG["block_size"])
            _, loss = current_model(x, y)
            losses.append(loss.item())
        values[name] = sum(losses) / len(losses)
    current_model.train()
    return values


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


signature = hashlib.sha256(
    f"{corpus_sha256}:{MODEL_CONFIG}:{MAX_STEPS}:{LEARNING_RATE}:{len(train_docs)}:{len(val_docs)}".encode()
).hexdigest()
last_path = ARTIFACTS / "chat_last.pt"
best_path = ARTIFACTS / "chat_best.pt"
history_path = ARTIFACTS / "chat_loss_history.json"
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
start_step, best_val, history = 0, float("inf"), []

if RESUME and last_path.exists():
    saved = torch.load(last_path, map_location=DEVICE, weights_only=True)
    if saved.get("signature") != signature:
        raise RuntimeError(
            f"The checkpoint in {ARTIFACTS} uses different data or settings. "
            "Move that folder aside before starting a new training run."
        )
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    start_step = saved["step"] + 1
    best_val = saved["best_val"]
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    print(f"Resuming at step {start_step}.")

for step in range(start_step, MAX_STEPS):
    model.train()
    x, y = get_batch(train_tokens, BATCH_SIZE, MODEL_CONFIG["block_size"])
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if (step + 1) % EVAL_INTERVAL == 0 or step + 1 == MAX_STEPS:
        measured = estimate_loss(model, train_tokens, val_tokens)
        history.append({"step": step + 1, **measured})
        atomic_json_save(history, history_path)
        if measured["validation"] < best_val:
            best_val = measured["validation"]
            atomic_torch_save(model.state_dict(), best_path)
        atomic_torch_save({
            "signature": signature,
            "step": step,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, last_path)
        print(
            f"chat {step + 1:>5}/{MAX_STEPS}: train loss {measured['train']:.3f}, "
            f"validation loss {measured['validation']:.3f}, "
            f"validation perplexity {math.exp(min(measured['validation'], 20)):.2f}"
        )

if not best_path.exists():
    raise RuntimeError("Training did not produce a checkpoint; check the log and available memory.")
model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
model.eval()

final_config = {
    **MODEL_CONFIG,
    "parameter_count": parameter_count,
    "end_token_id": end_token_id,
    "user_token_id": user_token_id,
    "assistant_token_id": assistant_token_id,
    "corpus_sha256": corpus_sha256,
    "from_scratch": True,
    "limitations": "Small model trained from scratch on English OpenAssistant conversations; not a dependable general assistant.",
}
temporary_tokenizer_path = TOKENIZER_PATH.with_suffix(".json.tmp")
tokenizer.save(str(temporary_tokenizer_path))
temporary_tokenizer_path.replace(TOKENIZER_PATH)
atomic_torch_save(model.state_dict(), ARTIFACTS / "model.pt")
atomic_json_save(final_config, ARTIFACTS / "config.json")
atomic_json_save(history, ARTIFACTS / "loss_history.json")

metrics = estimate_loss(model, train_tokens, val_tokens)
print(f"\nExported model to {ARTIFACTS / 'model.pt'}")
print(f"Validation loss: {metrics['validation']:.3f}")
print(f"Validation perplexity: {math.exp(min(metrics['validation'], 20)):.2f}")
sample = generate(
    model,
    tokenizer,
    USER_TOKEN + "Write a short scene between a king and a traveler.\n" + ASSISTANT_TOKEN,
    end_token_id,
    max_new_tokens=128,
    temperature=0.7,
    stop_token_ids=[user_token_id],
)
print("\nGenerated sample (not a verified factual answer):")
print(sample)

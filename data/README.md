# Training data

The chat-training notebook downloads `2023-04-12_oasst_ready.messages.jsonl.gz` from OpenAssistant OASST1 and stores it as `data/raw/oasst1_ready.messages.jsonl.gz`. It filters for English messages and excludes deleted, explicitly rejected, and marked-synthetic messages, then builds conversation paths using message parent links. The training script records a SHA-256 hash for the downloaded file and splits by conversation tree before training the tokenizer.

OASST1 contains human-generated, human-annotated assistant-style conversations and is published under the [Apache License 2.0](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE). The dataset card describes the message roles and conversation-tree structure: [OASST1 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1).

The training notebook downloads dataset text only. It does not download pretrained model weights or a pretrained tokenizer.


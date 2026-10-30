# Training data

Run `build_slm.py` to download and prepare the text corpora automatically. It downloads dataset text only. It does not download pretrained model weights or a pretrained tokenizer.

## WikiText-103

The pipeline downloads the raw WikiText-103 archive to `data/raw/wikitext-103-raw-v1.zip` and extracts its train, validation, and test text files under `data/processed/wikitext-103/`. The model trains on the train split and measures validation loss on the validation split; the test split is kept separate. WikiText-103 contains over 100 million tokens from verified Good and Featured Wikipedia articles and is licensed CC BY-SA 4.0. See the [dataset card](https://huggingface.co/datasets/Salesforce/wikitext).

## OpenAssistant OASST1

The pipeline uses `2023-04-12_oasst_ready.messages.jsonl.gz` at `data/raw/oasst1_ready.messages.jsonl.gz` (and downloads it if missing). It keeps English, non-deleted, non-synthetic messages that were not explicitly rejected, follows parent links to build conversation paths, and splits by conversation tree to avoid putting sibling paths in both training and validation. OASST1 contains human-generated assistant-style conversations and is licensed Apache 2.0. See the [dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1) and [license](https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/LICENSE).

Tokenizer training uses only the WikiText and OASST1 training portions. The test split is not used for training. The first run needs roughly 750 MB for WikiText plus about 35 MB for OASST1, with temporary space while downloading and extracting.


# Small language model research project

This project contains a small decoder-only language model initialized from random weights. `build_slm.py` downloads public **text datasets**, preprocesses them, trains a tokenizer from that text, then trains the model in two stages:

1. Language-model pretraining on WikiText-103.
2. Continued chat-style training on English OpenAssistant OASST1 conversations.

It does **not** download pretrained model weights or a pretrained tokenizer. Running the full script starts training; no training run is performed as part of this code change.

## Realistic goal

Treat the result as a learning and research prototype. With a model this small and these limited corpora, it may learn some language patterns and reply-shaped text, but it will not become a dependable general assistant. It can be incoherent, repeat text, and invent facts. Building a capable general assistant entirely from random weights requires far more suitable data and compute than this project or a home 4 GB GPU can provide. The practical goal here is to build and inspect a complete small-model pipeline.

## Hardware and time

The default model has about 2.6 million parameters, context length 128, and batch size 2. Its FP32 weights, gradients, and AdamW state use about 42 MB before activations and runtime overhead. For this exact small configuration, a CUDA GPU with 4 GB VRAM should be enough in normal conditions; if CUDA memory runs out, set `--batch-size 1`. This is a reasoned estimate, not a benchmark from a training run.

The project notes currently describe a GTX 1650 with 4 GB VRAM, about 16 GB system RAM, and a CPU-only PyTorch installation. In that setup, this code will use the **CPU** until a CUDA-enabled PyTorch build is installed. Confirm that `torch.cuda.is_available()` returns `True`; select the Windows and CUDA build that matches your machine on the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

| Model size | Persistent training memory estimate* | Practical guidance |
| --- | ---: | --- |
| This project, ~2.6M parameters | ~42 MB in FP32 | 4 GB VRAM is reasonable for the configured batch/context; CPU also works, much slower. |
| ~10M parameters | ~160 MB in FP32 | Several GB VRAM is generally comfortable for small batches. |
| ~100M parameters | ~1.6 GB in FP32 | Plan for roughly 8 GB or more after activations and runtime overhead. |
| ~1B parameters | ~16 GB in FP32 | Usually needs 24 GB+ VRAM or multiple GPUs, depending on context and batch. |

\*Approximation for FP32 parameters, gradients, and AdamW's two optimizer states: about 16 bytes per parameter, **before** activation memory. Longer context and larger batches can raise memory substantially. Mixed precision has a different estimate; the [Hugging Face memory guide](https://huggingface.co/docs/transformers/en/model_memory_anatomy) explains the components. Model quality also depends on training tokens and data quality, not just VRAM. Scaling research studies model size and training data together; see [Hoffmann et al.](https://arxiv.org/abs/2203.15556).

Expect the first data preparation run to need about 750 MB for WikiText archive plus extracted files, about 35 MB for OASST1, and additional space for the Python environment and model checkpoints. Keep at least 2 GB free. The default training preset is intentionally small for experimentation; CPU training may take a long time. No wall-clock estimate has been benchmarked on your machine.

## Install

Use Python 3.11 and a virtual environment. In PowerShell, from this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Those commands install CPU-only PyTorch. To use an NVIDIA GPU, install the matching CUDA-enabled PyTorch build using the official selector linked above, then confirm CUDA is available. The optional web-search checkbox in the app uses DDGS; install it separately with `python -m pip install ddgs`. Local chat does not require it.

## Prepare data and train

Run from the activated environment in the project folder:

```powershell
python build_slm.py --prepare-only
```

This downloads WikiText-103 and downloads OASST1 only if its compressed file is not already in `data/raw/`. It extracts the three WikiText splits, filters OASST1, builds conversation paths, makes a tree-level validation split, and trains a new tokenizer from the training portions. It stops before training the model.

To run both training stages and export a model:

```powershell
python build_slm.py
```

Or open `train_chat_from_scratch.ipynb` in Jupyter and run its cells in order. The notebook calls the same script. The script saves checkpoints after each evaluation interval, so rerunning it resumes the active stage. Default settings are 10,000 WikiText steps and 3,000 chat steps. Shorter trial run:

```powershell
python build_slm.py --pretrain-steps 100 --chat-steps 50
```

This short command is only for checking the pipeline; it is not enough to judge model quality. Model weights and checkpoints are saved under `artifacts/scratch_chat/`. The previous model under `artifacts/chat/` is preserved, and the app uses the new model once its exported files are available.

If you change the data or training settings after a run has started, move `artifacts/scratch_chat/` aside first to begin with fresh checkpoints.

## Run the chat app

After training exports `model.pt`, `config.json`, and `tokenizer.json`, run:

```powershell
python -m streamlit run app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`. Inference and chat history stay on the app host. The app does not execute generated code. Generated answers are not verified; if web search is enabled, review the linked sources directly.

## Datasets and licensing

- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext): over 100 million tokens from verified Wikipedia articles; CC BY-SA 4.0. The project uses the raw-v1 train, validation, and test split files.
- [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1): human-generated assistant-style conversations; Apache 2.0. The pipeline keeps English, non-deleted, non-synthetic messages not explicitly rejected, and splits by conversation tree.

See [data/README.md](data/README.md) for paths and preprocessing details. Review dataset terms and provenance before sharing or distributing any trained model.

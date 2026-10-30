# Local chat model trained from scratch

This project builds its language model from randomly initialized PyTorch weights. It does not download pretrained model weights or a pretrained tokenizer. The chat-training notebook downloads an open conversation **dataset** (text only), trains a new tokenizer, trains the model locally, and exports it for the Streamlit app.

Use `train_chat_from_scratch.ipynb` to train the model used by the chat app.

## What to expect

The chat model is a small research prototype trained from scratch on a limited public conversation corpus. It can produce simple conversational replies, but it will not match a general-purpose assistant. It can misunderstand requests, repeat itself, or make up facts. Check important answers independently.

The inspected computer has Python 3.11.9, 15.8 GiB RAM, a GTX 1650 with 4 GiB VRAM, and a CPU-only PyTorch build. The chat model uses 4 decoder blocks, width 128, context 256, and a 2,048-token target vocabulary. Training uses CPU unless a CUDA-enabled PyTorch build is installed. The notebook saves resumable checkpoints, but the full run can take a while.

## Install

Use Python 3.11 and a virtual environment. In PowerShell from this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The optional web-search checkbox uses DDGS. Install it separately with `python -m pip install ddgs`. Without it, local chat still works.

## Train the chat model

From the activated environment, start JupyterLab in the project folder:

```powershell
jupyter lab
```

Open `train_chat_from_scratch.ipynb` and run its cells in order. The first training run downloads the OASST1 conversation dataset, about 34 MB compressed, into `data/raw/`. This is training text only; no model weights are fetched. The notebook filters for English, non-deleted, non-synthetic messages and builds conversation paths. The tokenizer and model are initialized from scratch.

Training writes checkpoints and exports the app-ready model under `artifacts/chat/`. If training is interrupted, rerun the training cell; it resumes from its last checkpoint. If you change the training data or settings, move `artifacts/chat/` aside before starting a fresh run.

## Run the app

After the chat notebook exports the files under `artifacts/chat/`, run:

```powershell
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`. The model and chat history stay on the app host. Web search is optional; if enabled, the query is sent to a public search service and snippets are added as references. Review source links directly.

The app does not execute generated code. Treat generated text as unverified.

## Data and model artifacts

See [data/README.md](data/README.md) for the OASST1 source and filtering notes. OASST1 is licensed under Apache 2.0. Model checkpoints, downloaded data, and generated artifacts are ignored by Git. Review data provenance, licenses, and deployment constraints before sharing a trained model.

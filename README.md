# Tiny language model, trained locally

This project trains a small decoder-only Transformer from randomly initialized PyTorch weights in JupyterLab and serves the resulting checkpoint through Streamlit. It does not use Ollama, a hosted inference API, or any pretrained model or tokenizer. The tokenizer is trained from the training split with Hugging Face Tokenizers.

## Hardware fit and limitations

The inspected computer has Python 3.11.9, 15.8 GiB RAM (7.4 GiB available at inspection), an NVIDIA GTX 1650 with 4 GiB VRAM, and about 424 GiB free disk. The installed PyTorch is 2.14.1+cpu, so it cannot use the GPU. The notebook therefore uses CPU. The recommended model has 2 decoder blocks, width 128, 4 heads, context 128, and a target vocabulary of 512 tokens: about 478,720 parameters when the tokenizer reaches that vocabulary size. This is small enough to train on the available CPU and memory, but no training time or quality is promised.

The training data is a roughly 1.06 MB subset of Shakespeare's plays, with a separate small CC0 conversation set for instruction-format practice. This is an educational text generator, not a capable general assistant. It can produce archaic, repetitive, incoherent, or false text; the short context also limits how much conversation it can use. Search snippets can give it recent context, but do not make its answers reliable. Generated code is not verified, and the app never executes it.

## Recorded training run

The notebook was run on the inspected CPU-only PyTorch environment for 500 pretraining steps and 80 instruction-tuning steps. The last periodic pretraining evaluation recorded validation loss 4.683 (perplexity 108.04). Instruction tuning recorded 4.316 (perplexity 74.90) at step 80; a separately sampled final evaluation was 4.328 (perplexity 75.76) on four held-out conversations. That small validation set is noisy and does not measure general assistant ability. One generated answer to “What does a tokenizer do?” was “Asod.”, which illustrates the model's limitations. Samples can vary because generation is sampled.

## Install

Use Python 3.11 and a virtual environment. In PowerShell from this project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The optional web-search checkbox uses the real DDGS metasearch package. Install it separately with `python -m pip install ddgs`. If it is not installed, the chat still works and the app explains how to enable search.

The inspected GPU is not used by this CPU-only PyTorch build. If you want to try GPU training, install a CUDA-enabled PyTorch build selected for the local Windows driver using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). The GTX 1650 has only 4 GiB VRAM; keep the model and batch small. The notebook checks `torch.cuda.is_available()` and otherwise falls back to CPU.

## Train in JupyterLab

From the activated environment, start JupyterLab in the project folder:

```powershell
jupyter lab
```

Open `train_from_scratch.ipynb` and run its cells in order. The notebook downloads only the text corpus from the documented source into `data/raw/`; it records the corpus hash, deduplicates paragraphs, makes the held-out split before training a tokenizer, and trains the tokenizer on training text only. If the download is blocked, save the source text manually to `data/raw/tinyshakespeare.txt` and rerun the corpus cell.

Training saves atomic resumable checkpoints, configuration, tokenizer, and validation loss history under `artifacts/`. It reports validation loss and perplexity and generates samples from the trained weights. No metric or sample is pre-filled as a training result. The second training stage uses the openly licensed CC0 examples in `data/instructions_cc0.jsonl`; it is deliberately small and does not confer general assistant ability.

If training is interrupted, rerun the training cell; it resumes at the last saved evaluation checkpoint. If data or model settings change, the notebook reports that the checkpoint is incompatible. Move the ignored `artifacts/` folder aside only if you want to start a fresh run.

## Run the assistant

After the notebook exports `artifacts/model.pt`, `artifacts/config.json`, and `artifacts/tokenizer.json`:

```powershell
streamlit run app.py
```

Model inference and the chat transcript are kept in the Streamlit app session on the app host. Search is optional and separate: when enabled, the user's query is sent to DDGS, the app displays result links, and a few retrieved snippets are added to the local model prompt. Conversation text is not sent to a hosted language model.

The app reports missing or incompatible checkpoints clearly. Search can fail independently without stopping local generation. The model only sees a short recent context because its context window is 128 tokens.

## Hosting and artifact handling

For a private deployment, install the same Python dependencies on the host, place the locally trained `artifacts/` folder on that host, and launch Streamlit with `streamlit run app.py --server.address 0.0.0.0`. The host performs inference; CPU-only hosts may respond slowly, and concurrent users share the host's memory and compute. A GPU helps only when the host has a supported NVIDIA GPU and a matching CUDA-enabled PyTorch build. Public hosting makes the host operator responsible for access controls and the handling of chat sessions; web search also sends enabled queries to an external search service.

Model artifacts and downloaded training data are ignored by Git. Do not commit secrets or model files. If you later choose to publish or deploy a checkpoint, review its corpus provenance, license, size, and host limits first. This project does not upload or publish any artifacts.

## Data and references

See [data/README.md](data/README.md) for the corpus source and rights notes. The corpus is a small part of Shakespeare's public-domain works; the instruction examples are project-authored and dedicated to CC0 1.0.

The tokenizer workflow follows the official [Hugging Face Tokenizers training documentation](https://huggingface.co/docs/tokenizers/v0.20.3/en/training_from_memory). The decoder uses PyTorch's documented [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention) with an explicit causal mask. The chat interface uses Streamlit's [chat elements](https://docs.streamlit.io/develop/api-reference/chat). Optional search uses the published [DDGS Python package](https://pypi.org/project/ddgs/).

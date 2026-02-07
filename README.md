# Minimal LLaMA Inference Engine in Python

## Introduction

This project is a **from-scratch reimplementation of a LLaMA inference engine written entirely in Python**.  
It is a **pedagogical and experimental implementation** whose goal is to understand how LLaMA-style models perform inference internally, rather than achieving speed or efficiency.

The code is capable of running **LLaMA-2-7B Chat quantized in Q8_0 (GGUF format)** without relying on:

- `llama.cpp`
- `llama-cpp-python`
- any existing inference backend

All major components are implemented explicitly: GGUF parsing, Q8_0 dequantization, tokenization, transformer blocks, attention, and sampling.

---

## Installation (Linux)

### Environment setup

Install the files in a directory `myFolder`.  
Using a virtual environment is recommended.

```bash
cd myFolder
python3 -m venv .
source bin/activate
````

### Run the program

```bash
python -m main
```

### First run: model installation

On the first run, the program will ask whether you want to install a language model.
If accepted, it downloads:

* **LLaMA-2-7B-Chat Q8_0 (GGUF)** from Hugging Face

The model is then decompressed and stored as pickle files.

After the installation completes, **restart the program**.

---

## Usage

A simple Qt-based graphical interface opens.

* Enter a prompt in the input field
* Token generation is performed sequentially
* Generated tokens are printed in the terminal before appearing in the UI

Because the model may have difficulty emitting a correct stop sequence, the response length is limited by:

```python
MAX_TOKENS = 200
```

This constant is defined at the beginning of `main.py` and can be adjusted.

---

## Project structure

```
.
├── main.py            # GUI and generation loop
├── tokenizer.py       # Minimal BPE-style tokenizer
├── loader.py          # GGUF loader and model installation
├── decompression.py   # Q8_0 dequantization
├── transformer.py    # LLaMA forward pass implementation
├── check.py           # Debug / validation utility
└── models/            # GGUF model and generated pickle files
```

---

## Technical overview

### 1. Model loading (`loader.py`)

* Direct parsing of the GGUF file format
* Extraction of:

  * model metadata and hyperparameters
  * tensor descriptors and raw data
* Dequantization of tensors
* Serialization to pickle files to avoid repeated decoding

---

### 2. Dequantization (`decompression.py`)

* Manual implementation of **Q8_0** dequantization
* Conversion of:

  * `int8` values multiplied by a per-block `fp16` scale
  * into `float32`
* Explicit fp16 → fp32 conversion

---

### 3. Tokenization (`tokenizer.py`)

* Simple bigram-based merge tokenizer
* Uses the tokenizer vocabulary embedded in the GGUF file
* Explicit insertion of LLaMA-2 chat special tokens

---

### 4. Transformer inference (`transformer.py`)

Explicit NumPy implementation of the LLaMA architecture:

* RMSNorm
* Rotary Positional Encoding (RoPE)
* Multi-head self-attention with KV cache
* SwiGLU feed-forward networks
* Residual connections
* Final projection and stochastic token sampling

All computations are performed on CPU using NumPy.

---

### 5. Generation loop and UI (`main.py`)

* Sequential token-by-token inference
* Manual management of past key/value states
* Simple stop-token logic
* Basic Qt graphical interface

---

## Known limitations

* Very slow (pure Python + NumPy)
* High memory usage
* No GPU support
* No batching
* Basic sampling strategy
* Minimal tokenizer implementation

These limitations are intentional and reflect a deliberate focus on code readability and conceptual clarity rather than performance.

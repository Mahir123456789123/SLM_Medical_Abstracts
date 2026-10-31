# Small Language Model from Scratch on PubMed Abstracts

A ~30M-parameter, decoder-only GPT-style language model, built and trained entirely from scratch on biomedical research abstracts from PubMed. No pretrained weights are used anywhere in the pipeline: data cleaning, tokenisation, model definition, training loop and text generation are all implemented in this repository.

The goal is not to compete with large general-purpose models. It is to build a small, transparent, domain-specific model that is cheap enough to train on a single consumer GPU, and to study what such a model does and does not learn about biomedical language.

> **Not for medical use.** This model generates statistically plausible biomedical-sounding text. It does not understand medicine, and its output can be wrong, incoherent or contain invented numbers. Do not use it for clinical decisions, medical advice or as a source of facts.

---

## Table of contents

1. [Highlights](#highlights)
2. [Dataset](#dataset)
3. [Model architecture](#model-architecture)
4. [Training setup](#training-setup)
5. [Results](#results)
6. [Sample generations](#sample-generations)
7. [What the model learned, and where it fails](#what-the-model-learned-and-where-it-fails)
8. [Repository contents](#repository-contents)
9. [How to run](#how-to-run)
10. [Possible improvements](#possible-improvements)
11. [Acknowledgements and licence](#acknowledgements-and-licence)

---

## Highlights

- **From scratch.** A GPT-style transformer, closely following the [nanoGPT](https://github.com/karpathy/nanoGPT) design, trained from random initialisation.
- **Domain-specific corpus.** 37,138 unique PubMed abstracts (about 5.9 million words) across seven biomedical and neuroscience topics.
- **Small and trainable locally.** 6 layers, 6 heads, 384-dimensional embeddings, 128-token context, about 30M parameters. Trained for 50,000 iterations on a consumer GPU using mixed precision and gradient accumulation.
- **Honest evaluation.** Final train loss 5.37 and best validation loss 5.65. The model picks up biomedical vocabulary and the structure of an abstract, but its output is not factually reliable (see [limitations](#what-the-model-learned-and-where-it-fails)).

---

## Dataset

**Source:** a CSV of PubMed paper titles, links and abstracts (`pubmed_dataset/pubmed_abstracts.csv`), 13,200 rows across 7 topical columns.

**Topics covered:**

- Deep learning in medicine
- COVID-19
- Human connectome mapping
- Virtual reality
- Brain-machine / brain-computer interfaces
- Electroactive polymers and PEDOT electrodes
- Neuroprosthetics

### Preprocessing (`dataset.py`)

The raw CSV needed more than a plain `read_csv`:

1. **Parsing.** Each abstract cell is a Python-serialised tuple, e.g. `(['sentence one', 'sentence two'], 'paper title')`. It is parsed back into an object with `ast.literal_eval()` and the sentences are joined into one paragraph.
2. **Deduplication.** Papers can appear under more than one topic, so duplicates are removed with order-preserving logic.
3. **Cleaning.** URLs are stripped with a regex, whitespace is normalised, and HTML artefacts are removed.
4. **Filtering.** Abstracts shorter than 20 words are discarded.

The cleaned abstracts are written to a single text file, `pubmed_corpus.txt`.

### Corpus statistics

| Statistic | Value |
|---|---|
| Unique abstracts | 37,138 |
| Total words | 5,907,395 |
| Characters | 41,248,384 |
| Average abstract length | about 159 words |

### Tokenisation

The corpus is tokenised with the **GPT-2 BPE** tokeniser (vocabulary size 50,257) and written to `train.bin` and `validation.bin` as binary arrays of token ids. These files are produced by the notebook.

---

## Model architecture

A decoder-only transformer with pre-normalisation.

| Hyperparameter | Value |
|---|---|
| Transformer blocks (`n_layer`) | 6 |
| Attention heads (`n_head`) | 6 |
| Embedding dimension (`n_embd`) | 384 |
| Context window (`block_size`) | 128 tokens |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Feed-forward width | 1,536 (4x expansion), GELU |
| Dropout | 0.1 |
| Parameters | about 30M |

Design details:

- **Embeddings:** learned token (`wte`) and positional (`wpe`) embeddings, summed and passed through dropout.
- **Attention:** multi-head causal self-attention with a fused QKV projection. Uses Flash Attention (`F.scaled_dot_product_attention`) when available.
- **Feed-forward:** two-layer MLP, 384 -> 1536 -> 384, with GELU and dropout on the output projection.
- **Normalisation:** custom LayerNorm applied before each sub-layer (pre-norm).
- **Weight tying:** the token embedding matrix and the output `lm_head` share weights.
- **Initialisation:** normal with mean 0 and std 0.02; output projections (`c_proj`) are additionally scaled by `1/sqrt(2 * n_layer)` to keep the residual stream stable at initialisation.

---

## Training setup

| Parameter | Value |
|---|---|
| Iterations | 50,000 |
| Batch size (per step) | 8 |
| Gradient accumulation steps | 128 |
| Effective batch | 1,024 sequences of 128 tokens per update |
| Optimiser | AdamW (betas 0.9 / 0.95, eps 1e-9, weight decay 0.1) |
| Peak learning rate | 1e-4 |
| LR schedule | 2,000-step linear warm-up, then cosine decay |
| Gradient clipping | max norm 0.5 |
| Precision | float16 autocast with `GradScaler` |
| Evaluation | every 500 iterations |
| Checkpointing | best validation loss kept, previous best retained as fallback |

### Differences from the reference setup

The pipeline follows a reference small-language-model recipe (originally trained on the TinyStories dataset) with these changes:

| Change | Reference | This project | Why |
|---|---|---|---|
| Batch size | 32 | 8 | Fit consumer-GPU memory |
| Gradient accumulation | 32 | 128 | Keep the effective batch the same (1,024 sequences per update) |
| Iterations | 20,000 | 50,000 | Biomedical text is harder than simple stories; loss was still falling |
| Warm-up steps | 1,000 | 2,000 | Scaled with the longer run |
| Mixed precision | bfloat16 | float16 + GradScaler | Consumer GPUs often lack bfloat16 support |
| Checkpoints | best only | best plus previous best, stale ones deleted | Disk space and recovery over a long run |

The model architecture itself was left unchanged.

---

## Results

| Metric | Value |
|---|---|
| Initial loss | about 10.4 |
| Final training loss | **5.3726** |
| Best validation loss | **5.6535** (iteration 49,500) |
| Train / validation gap | about 0.28 |

<!-- Add the loss curve to the repo (e.g. assets/loss_curve.png) and uncomment:
![Training and validation loss](assets/loss_curve.png)
-->

Training and validation loss fall smoothly from about 10.4 with no divergence or instability, and the curves stay close together, with validation loss flattening slightly more than training loss late in the run. The curve was still descending at 50,000 iterations, so the model was limited by training budget and data rather than by overfitting.

A validation loss of 5.65 is still high in absolute terms, which matches the quality of the generations below. No quantitative metric beyond loss (such as perplexity on a separate test set or BERTScore) has been computed yet; evaluation so far is qualitative.

---

## Sample generations

Full samples after 50,000 iterations are in [`output_50000.txt`](output_50000.txt). Short extracts:

**Prompt:** `MRI imaging results suggest`

> MRI imaging results suggest that our data from the potential to the brain network connectome. To address this study, we investigated the main objective of a novel model using a novel method... We propose a deep learning method for the deep network with a deep learning model. In this paper, we present a model to assess the network... Results show that the two types of the use of the proposed method achieved...

**Prompt:** `A randomized controlled trial of`

> A randomized controlled trial of a virtual reality. A virtual reality simulator as the VR was also to be an alternative to reduce the system. The proposed method was also associated with the virtual reality...

**Prompt:** `Patients diagnosed with type 2 diabetes`

> Patients diagnosed with type 2 diabetes.0.7% and 0.8% for the DNN of the accuracy of 0.2. 0. The proposed method is an emerging approach (p=0.0. 0.0.15.3) and 0...

**Prompt:** `The effects of COVID-19 on`

> The effects of COVID-19 on COVID-19. It is a novel coronavirus (PEDOT) in SARS-19, and its effect of a 2 (SARS-2) were used to be considered for COVID-CoV-19 patients...

These outputs are not factual statements. They show what a small model trained on this corpus actually produces.

---

## What the model learned, and where it fails

**What it learned**

- **Domain vocabulary.** Outputs stay inside the biomedical register: *coronavirus*, *SARS-CoV-2*, *connectome*, *EEG*, *deep learning*, *virtual reality*, *BCI*.
- **The shape of an abstract.** It reproduces the typical arc of a scientific abstract (problem, "we propose", method, evaluation, "results show") purely from exposure to the text.
- **Local fluency.** Short phrases and clauses are often plausible.

**Where it fails**

- **Global incoherence.** With a 128-token context and about 30M parameters, sentences drift and do not build on one another.
- **Repetition.** Phrases such as "the proposed method" and "deep learning" loop heavily.
- **Cross-topic bleeding.** Terms from one topic appear in another, e.g. *PEDOT* (an electrode polymer) in a COVID-19 completion. The model treats the seven topics as one shared vocabulary pool.
- **Numerical hallucination.** It has learned that papers report accuracies and p-values, but not how numbers are formatted or what they mean, so it produces malformed figures like `p=0.0. 0.0.15.3`.
- **No instruction following.** It is a pure text-completion model: it cannot answer questions or summarise.

---

## Repository contents

```
.
|-- Vizuara_AI_Labs_Small_Language_Model_Scratch_Final.ipynb   Main notebook: tokenisation, model, training, evaluation, generation
|-- dataset.py                  Cleans pubmed_abstracts.csv into pubmed_corpus.txt
|-- pubmed_dataset/
|   `-- pubmed_abstracts.csv    Raw dataset (titles, links, abstracts)
|-- pubmed_corpus.txt           Cleaned, combined abstract corpus
|-- train.bin                   Tokenised training split
|-- validation.bin              Tokenised validation split
|-- output_50000.txt            Sample generations after 50,000 iterations
|-- Build SLM.pdf               Assignment brief for the project
`-- README.md
```

---

## How to run

### 1. Environment

Python 3.10 or newer and a CUDA-capable GPU are recommended (training on CPU is impractical). Install the libraries imported at the top of the notebook; typically:

```bash
pip install torch numpy pandas tiktoken matplotlib tqdm
```

### 2. Rebuild the text corpus (optional)

`pubmed_corpus.txt` is already included. To regenerate it from the raw CSV:

```bash
python dataset.py
```

This reads `pubmed_dataset/pubmed_abstracts.csv`, parses and deduplicates the abstracts, strips URLs and extra whitespace, drops abstracts under 20 words, and writes `pubmed_corpus.txt`.

### 3. Tokenise, train and generate

Open `Vizuara_AI_Labs_Small_Language_Model_Scratch_Final.ipynb` (Jupyter, VS Code or Colab) and run the cells in order. The notebook:

1. Tokenises the corpus with the GPT-2 BPE tokeniser and writes `train.bin` and `validation.bin`.
2. Builds the model and training configuration.
3. Trains for 50,000 iterations, evaluating every 500 and saving the best checkpoint.
4. Plots the loss curves.
5. Generates text from prompts such as `The effects of COVID-19 on`.

On a consumer GPU, if you hit out-of-memory errors, lower the per-step batch size and raise the gradient accumulation steps by the same factor so the effective batch stays the same.


## Acknowledgements and licence

- Model design follows [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
- The training notebook is based on the Vizuara AI Labs small-language-model-from-scratch walkthrough.
- Text comes from [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts. Check PubMed's terms of use before redistributing the data.

> **TODO:** add the dataset's original source (e.g. Kaggle link) and a `LICENSE` file for the code, and name it here.

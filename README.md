# SemRAG: Scene-Graph Guided Retrieval-Augmented Image Captioning with Gated Multi-Token Fusion and Structured Knowledge Base

> 🌐 **Language / Ngôn ngữ:** **English** | [🇻🇳 **Bấm vào đây để xem toàn bộ nội dung bằng Tiếng Việt**](README_VI.md)

[![Language](https://img.shields.io/badge/Language-Ti%E1%BA%BFng%20Vi%E1%BB%87t-blue.svg)](README_VI.md)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU%201.7%2B-blue.svg)](https://github.com/facebookresearch/faiss)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-purple.svg)](https://github.com/openai/CLIP)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19%20%7C%20Vite-61dafb.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Official Research Repository** for the Graduation Thesis in Information Technology at **Ho Chi Minh City University of Industry and Trade (HUIT)**.  
> **Authors (Group 02):** Ngo Quoc Binh (2001230056), Nguyen Thi Quyen (2001230745), Nguyen Quoc Viet (2001231038)  
> **Supervisors:** Dinh Thi Man, M.Sc. & Nguyen Van Thinh, Ph.D.  
> **Evaluation Period:** November 2026  

---

## 📖 Table of Contents
- [1. Overview & Core Motivation](#1-overview--core-motivation)
- [2. System Architecture](#2-system-architecture)
  - [Block A: Input Multimodal Encoding](#block-a-input-multimodal-encoding)
  - [Block B: Hybrid Reranking & Diversity Selection (MMR)](#block-b-hybrid-reranking--diversity-selection-mmr)
  - [Block C: Multi-Layer Structured Knowledge Base (SemKB)](#block-c-multi-layer-structured-knowledge-base-semkb)
  - [Block D: Gated Multi-Token Fusion Decoder](#block-d-gated-multi-token-fusion-decoder)
- [3. Key Benchmark Results](#3-key-benchmark-results)
  - [MS COCO 2014 Benchmark (Model Progression across 3 Seeds)](#ms-coco-2014-benchmark-model-progression-across-3-seeds)
  - [Flickr30k Evaluation (In-Domain Option A vs. Zero-Shot Option B)](#flickr30k-evaluation-in-domain-option-a-vs-zero-shot-option-b)
  - [Comprehensive Ablation Study](#comprehensive-ablation-study)
  - [Modality Isolation (Visual-Only vs. Retrieval-Only Mode)](#modality-isolation-visual-only-vs-retrieval-only-mode)
  - [Parameter Sensitivity Analysis ($K$ and $M$)](#parameter-sensitivity-analysis-k-and-m)
- [4. Interactive Web Application](#4-interactive-web-application)
- [5. Repository Structure & Notebook Index](#5-repository-structure--notebook-index)
- [6. Installation & Reproduction Guide](#6-installation--reproduction-guide)
  - [Environment Setup](#environment-setup)
  - [Downloading Model Checkpoints & Knowledge Base from Google Drive](#downloading-model-checkpoints--knowledge-base-from-google-drive)
  - [Running Evaluation Notebooks](#running-evaluation-notebooks)
  - [Launching the Web Application](#launching-the-web-application)
- [7. Citation & BibTeX](#7-citation--bibtex)
- [8. License & Acknowledgments](#8-license--acknowledgments)

---

## 1. Overview & Core Motivation

Standard image captioning systems rely exclusively on direct visual features extracted by deep Convolutional Neural Networks (CNN) or Vision Transformers (ViT). While achieving reasonable surface fluency, they frequently suffer from two critical limitations:
1. **Factual Hallucinations:** Inventing non-existent objects or hallucinating erroneous attributes due to language prior biases.
2. **Generic, Dull Captions:** Producing repetitive descriptions that fail to convey nuanced interactions and contextual relationships.

Retrieval-Augmented Generation (RAG) mitigates these bottlenecks by conditioning generation on external knowledge retrieved from verified human-annotated corpora. However, conventional RAG systems (Old SemKB) face severe obstacles:
* **Duplicate candidate bias:** Lexically redundant captions flood the nearest-neighbor search.
* **Surface lexical matching:** Simple cosine or unigram matching overlooks structured syntactic roles (Subject–Predicate–Object).
* **Mean-pooling representation bottleneck:** Compressing multiple retrieved candidates into a single averaged vector loses individual descriptive nuances.

### The SemRAG Solution
To resolve these challenges, **SemRAG** introduces:
1. **Multi-Layer Structured Knowledge Base (SemKB):** Augments raw captions with canonical deduplication flags (`duplicate_of`), standardized object entities, explicit RelTR Subject–Predicate–Object (`SPO`) relation triplets, actions, attributes, and corpus-wide 1–4 gram TF-IDF consensus weights.
2. **Scene-Graph-Guided Hybrid Reranking & MMR:** Combines global CLIP visual-semantic similarity with explicit SPO relation alignment, filtered through Maximal Marginal Relevance (MMR, $\lambda=0.7$) to eliminate redundancy.
3. **Dynamic Gated Multi-Token Fusion Decoder:** Projects Top-$K$ retrieved candidates as independent memory slots ($K \times 256$) and employs an adaptive gating mechanism ($g \in [0, 1]$) to balance visual observation and external knowledge dynamically.

---

## 2. System Architecture

The overall architecture consists of four collaborative blocks as illustrated in `Mô hình kho tri thức mới.pdf`:

```
+---------------------------------------------------------------------------------------------+
|                                    SEMRAG ARCHITECTURE                                      |
+---------------------------------------------------------------------------------------------+
|                                                                                             |
|   +-----------------------+                         +-----------------------------------+   |
|   |   BLOCK A: INPUT      |                         |   BLOCK C: MULTI-LAYER KNOWLEDGE  |   |
|   |                       |                         |            BASE (SemKB)           |   |
|   |  - Query Image        |                         |                                   |   |
|   |  - CLIP ViT-B/32      |                         |  - 566,747 Multimodal Records     |   |
|   |    Embedding (512d)   |                         |  - Canonical Deduplication Flag   |   |
|   |  - RelTR Scene Graph  |                         |  - Normalized Entities (Objects)  |   |
|   |    Extraction (M=10)  |                         |  - Explicit SPO Relations         |   |
|   +-----------+-----------+                         |  - Actions & Attributes Metadata  |   |
|               |                                     |  - TF-IDF 1..4-gram Consensus     |   |
|               | (Visual Vector & Query SPO)         +-----------------+-----------------+   |
|               v                                                       |                     |
|   +-----------+-------------------------------------------------------+                 |   |
|   |   BLOCK B: RETRIEVAL, EXPANSION & SELECTION                                         |   |
|   |                                                                                     |   |
|   |  1. FAISS IndexFlatIP Dense Retrieval (Top-N = 50 candidate captions)               |   |
|   |  2. Semantic Expansion via SPO Neighborhood Consensus                               |   |
|   |  3. Hybrid Reranking Score:                                                         |   |
|   |     S_rel = 0.6 * S_CLIP + 0.4 * S_SPO_Sem                                          |   |
|   |  4. Maximal Marginal Relevance (MMR, lambda = 0.7) Selection -> Top-K = 5 Candidates |   |
|   |  5. Multi-Prototype SPO Triplet Aggregation                                         |   |
|   +-----------------------------------+-------------------------------------------------+   |
|                                       |                                                     |
|                                       | (K=5 Retrieved Vectors & Aggregated Context)        |
|                                       v                                                     |
|   +-----------------------------------+-------------------------------------------------+   |
|   |   BLOCK D: GATED MULTI-TOKEN FUSION DECODER                                         |   |
|   |                                                                                     |   |
|   |  - Multi-Token Memory Projection: M_knw in R^(B x 5 x 256)                          |   |
|   |  - Visual Memory Projection:      M_vis in R^(B x 1 x 256)                          |   |
|   |  - GatedFusion Module (tau = 1.0):                                                  |   |
|   |     g = sigmoid(Linear([v_norm; k_agg_norm; conf]))                                 |   |
|   |  - 3-Layer Dual-Branch Cross-Attention:                                             |   |
|   |     C_fused = (1 - g) * C_vis + g * C_knw                                           |   |
|   |  - Beam Search Decoder (Beam=3, Length Penalty alpha=0.6, max_len=30)               |   |
|   +-------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------+
```

### Block A: Input Multimodal Encoding
* **Global Visual Feature:** The input image $I$ is encoded into a 512-dimensional normalized embedding $v_{\text{img}} \in \mathbb{R}^{512}$ using frozen `CLIP ViT-B/32`.
* **Scene Graph Triplet Extraction:** `RelTR` (Scene Graph Transformer) extracts up to $M = 10$ high-confidence Subject–Predicate–Object relation triplets $G(I) = \{(s_m, p_m, o_m)\}_{m=1}^M$.

### Block B: Hybrid Reranking & Diversity Selection (MMR)
* **Initial Dense Query:** Queries FAISS (`IndexFlatIP`) to retrieve $N = 50$ candidate captions based on inner-product similarity $\tilde{S}_{\text{CLIP}}(I, c_i)$.
* **Semantic SPO Reranking:** Computes semantic compatibility $S_{\text{SPO-Sem}}$ between the image query scene graph and the pre-extracted candidate triplets in SemKB. The hybrid score is defined as:
  $$S_{\text{rel}}(I, c_i) = w \cdot \tilde{S}_{\text{CLIP}}(I, c_i) + (1 - w) \cdot S_{\text{SPO-Sem}}(G(I), G_{\text{KB}}(c_i)), \quad w = 0.6$$
* **Maximal Marginal Relevance (MMR):** Selects $K = 5$ diverse, non-redundant candidates:
  $$\text{MMR} = \arg\max_{c_i \in C \setminus S} \left[ \lambda S_{\text{rel}}(I, c_i) - (1 - \lambda) \max_{c_j \in S} \text{Sim}(c_i, c_j) \right], \quad \lambda = 0.7$$
  where $\text{Sim}(c_i, c_j) = 0.65 \cdot \text{Sim}_{\text{CLIP}} + 0.35 \cdot \text{Jaccard}_{\text{SPO}}$.

### Block C: Multi-Layer Structured Knowledge Base (SemKB)
| Specification | MS COCO 2014 (`knowledge_base_level4`) | Flickr30k (`knowledge_base_level4_Flick30K`) |
| :--- | :--- | :--- |
| **Total Split Records** | **566,747 entries** (Train + Restval, 113,287 images) | **145,000 entries** (Train, 29,000 images) |
| **Vector Index** | CLIP ViT-B/32 (512d, L2-normalized, `IndexFlatIP`) | CLIP ViT-B/32 (512d, L2-normalized, `IndexFlatIP`) |
| **Canonical Deduplication (`duplicate_of`)** | **21,916 duplicate captions** mapped to root IDs | **524 duplicate captions** mapped to root IDs |
| **Normalized Entities (`objects`)** | **566,671 records (99.99%)** | **144,996 records (100.0%)** |
| **SPO Relations (`relations`)** | **521,602 records (92.03%)** | **139,714 records (96.35%)** |
| **Actions & Attributes** | Explicit actions & attribute tokens | Actions: 88.67%, Attributes: 49.09% |
| **N-gram Consensus** | Corpus-wide TF-IDF 1..4-gram weights | Corpus-wide TF-IDF 1..4-gram weights |

### Block D: Gated Multi-Token Fusion Decoder
* **Multi-Token Representation:** Retains all $K = 5$ retrieved vectors without averaging:
  $$M_{\text{knw}} \in \mathbb{R}^{B \times 5 \times 256}$$
* **Dynamic Gated Fusion:** A lightweight neural gate predicts scalar balance coefficient $g \in [0, 1]$:
  $$g = \sigma\left(W_g [v_{\text{norm}}; k_{\text{agg\_norm}}; \text{conf}] + b_g\right)$$
* **Dual-Branch Cross-Attention:**
  $$C_{\text{fused}} = (1 - g) \odot \text{CrossAttn}(Q, M_{\text{vis}}) + g \odot \text{CrossAttn}(Q, M_{\text{knw}})$$
* **Inference:** Beam search (`beam_size = 3`, length penalty $\alpha = 0.6$, maximum length 30 tokens).

---

## 3. Key Benchmark Results

All experiments strictly adhere to a **100% controlled benchmarking protocol**:
- Fixed checkpoint weights (`best_model.pt`, 12,538,968 parameters, verified SHA-256 integrity).
- Fixed vocabulary (`vocab.json`, 16,214 words on COCO; 9,546 on Flickr30k).
- Evaluated across **3 standard independent seeds: `[42, 44, 46]`** on standard 5,000 Karpathy test images (COCO) and 1,000 test images (Flickr30k).

### MS COCO 2014 Benchmark (Model Progression across 3 Seeds)

Results reported as $\text{Mean} \pm \text{SD}$ on MS COCO 2014 Karpathy Test split (5,000 images / 25,010 references, evaluated via official `SafePTBTokenizer`):

| Model & Configuration | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr (SemKB) | CIDEr (Old SemKB) | $\Delta$ CIDEr | SPICE (SemKB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Visual-Only Baseline)** (Beam=3, No KB) | 74.21±0.19 | 57.80±0.21 | 44.12±0.17 | 33.68±0.14 | 26.77±0.10 | 54.99±0.13 | **107.15 ± 0.74** | 107.15 | 0.00 | 19.73 ± 0.10 |
| **WF-RAG (Basic RAG)** (CLIP Top-5 Mean-Pool) | 74.92±0.10 | 58.54±0.04 | 44.78±0.02 | 34.34±0.02 | 27.41±0.04 | 55.84±0.01 | **112.52 ± 0.34** | 107.57 | **+4.95** | 20.16 ± 0.16 |
| **RR-RAG (Scene-Graph Reranked RAG)** (RelTR SPO) | 74.97±0.21 | 58.54±0.15 | 45.00±0.10 | 34.71±0.09 | 27.68±0.02 | 55.95±0.05 | **114.13 ± 0.28** | 110.64 | **+3.49** | 20.68 ± 0.03 |
| **GM-RAG (Gated Multi-Token RAG)** (5 Tokens, No MMR)| 74.83±0.07 | 58.43±0.06 | 44.96±0.08 | 34.73±0.08 | 27.82±0.01 | 55.93±0.01 | **114.81 ± 0.20** | 110.97 | **+3.84** | 20.67 ± 0.04 |
| **SemRAG (Proposed Full Model)** (RelTR+MMR+Gated) | **74.92±0.05** | **58.57±0.03** | **45.13±0.04** | **34.90±0.06** | **27.85±0.02** | **56.01±0.01** | **115.41 ± 0.07** | 111.63 | **+3.78** | **20.76 ± 0.02** |

* Per-seed breakdown for SemRAG: Seed 42 = **115.38**, Seed 44 = **115.36**, Seed 46 = **115.49** CIDEr.
* Hallucination metrics on MS COCO Test: $\text{CHAIR}_s = \mathbf{8.21 \pm 0.08\%}$, $\text{CHAIR}_i = \mathbf{7.18 \pm 0.06\%}$.

---

### Flickr30k Evaluation (In-Domain Option A vs. Zero-Shot Option B)

Evaluated on Flickr30k 1,000 Karpathy test images across Seeds `[42, 44, 46]`:

#### Option A: In-Domain Flickr30k Evaluation (Trained on Flickr30k + Flickr30k SemKB)
| Model & Configuration | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr (SemKB) | $\Delta$ vs VOB | SPICE | $\text{CHAIR}_s \downarrow$ (%) | $\text{CHAIR}_i \downarrow$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Visual-Only Baseline)** | 68.85±0.38 | 25.89±0.24 | 21.10±0.13 | 48.14±0.16 | **53.85 ± 0.62** | 0.00 | 14.82±0.19 | 13.27±0.38 | 9.64±0.25 |
| **WF-RAG (Basic RAG)** | 67.66±0.22 | 25.21±0.09 | 21.44±0.02 | 48.20±0.12 | **58.29 ± 0.22** | +4.44 | 15.27±0.04 | 11.40±0.10 | 8.03±0.06 |
| **RR-RAG (Scene-Graph Reranked RAG)**| 69.97±0.66 | 26.44±0.29 | 21.90±0.05 | 49.02±0.11 | **59.53 ± 0.18** | +5.68 | 15.63±0.05 | 10.57±0.12 | 7.68±0.21 |
| **GM-RAG (Gated Multi-Token RAG)**| 70.89±0.15 | 26.95±0.08 | 21.92±0.06 | 49.20±0.14 | **60.70 ± 0.18** | +6.85 | 15.68±0.05 | 9.63±0.06 | 7.19±0.03 |
| **SemRAG (Proposed Full Model)**| **71.04±0.18** | **27.26±0.07** | **22.03±0.06** | **49.37±0.10** | **61.98 ± 0.21** | **+8.13** | **15.77±0.01** | **8.63 ± 0.06** | **6.47 ± 0.05** |

#### Option B: Zero-Shot Cross-Domain Transfer (COCO Model + COCO SemKB directly evaluated on Flickr30k)
| Model & Configuration | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | $\Delta$ vs VOB | SPICE | $\text{CHAIR}_s \downarrow$ (%) | $\text{CHAIR}_i \downarrow$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (COCO Baseline)** | 58.79±0.01 | 17.12±0.03 | 17.23±0.03 | 41.02±0.05 | **37.83 ± 0.14** | 0.00 | 10.95±0.04 | 36.10±0.10 | 23.29±0.03 |
| **WF-RAG (COCO Basic RAG)** | 59.15±0.46 | 17.41±0.23 | 17.75±0.22 | 41.69±0.31 | **41.51 ± 0.99** | +3.68 | 11.35±0.14 | 26.80±0.10 | 17.48±0.10 |
| **RR-RAG (COCO RelTR Rerank)**| 59.73±0.18 | 18.16±0.03 | 17.98±0.11 | 42.08±0.20 | **42.93 ± 0.56** | +5.10 | 11.81±0.10 | 24.40±0.10 | 16.07±0.04 |
| **GM-RAG (COCO Gated Multi-Token)** | 60.07±0.27 | 18.22±0.25 | 17.98±0.09 | 42.17±0.12 | **43.36 ± 0.24** | +5.53 | 11.92±0.11 | 22.10±0.10 | 14.84±0.07 |
| **SemRAG (COCO Proposed Model)** | **60.77±0.08** | **19.00±0.11** | **18.12±0.04** | **42.71±0.05** | **44.88 ± 0.21** | **+7.05 (+18.6%)**| **12.08±0.00** | **19.80 ± 0.10** | **13.82 ± 0.03** |

> **Key Transfer Takeaway:** Under zero retraining and zero domain fine-tuning, SemRAG boosts CIDEr by **+7.05 points (+18.64%)** and drastically cuts sentence hallucination rate ($\text{CHAIR}_s$) from **36.10% down to 19.80% (a 16.30 percentage point reduction)**.

---

### Comprehensive Ablation Study

Isolated on MS COCO 2014 Karpathy Test (5,000 images, 3 Seeds `[42, 44, 46]`):

| Ablation Configuration | Notebook Identifier | BLEU-4 | METEOR | CIDEr (SemKB) | $\Delta$ CIDEr | SPICE (SemKB) | $\Delta$ SPICE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SemRAG (Proposed Full Model)** | Proposed Pipeline | **34.90 ± 0.06** | **27.85 ± 0.02** | **115.41 ± 0.07** | **0.00** | **20.76 ± 0.02** | **0.00** |
| $\text{w/o MMR Diversity}$ | `Ablation_01 (no_mmr)` | 34.83 ± 0.05 | 27.82 ± 0.02 | **114.70 ± 0.11** | **-0.72** | 20.66 ± 0.04 | **-0.10** |
| $\text{w/o WordNet / SPO Expansion}$ | `Ablation_02 (no_semantic_expansion)` | 34.63 ± 0.11 | 27.74 ± 0.03 | **113.95 ± 0.14** | **-1.47** | 20.49 ± 0.02 | **-0.27** |
| $\text{w/o Multi-Token Memory}$ (Mean-Pool)| `Ablation_04A (no_multitoken_only)` | 34.83 ± 0.09 | 27.75 ± 0.02 | **113.80 ± 0.05** | **-1.61** | 20.58 ± 0.03 | **-0.18** |
| $\text{w/o Dynamic Gated Fusion}$ ($g=0.5$)| `Ablation_04B (no_gate_only)` | 34.39 ± 0.05 | 27.62 ± 0.01 | **113.00 ± 0.17** | **-2.41** | 20.42 ± 0.01 | **-0.34** |
| $\text{w/o Scene-Graph RelTR Reranking}$ | `Ablation_03 (no_reltr_reranking)` | 34.28 ± 0.17 | 27.60 ± 0.05 | **112.08 ± 0.33** | **-3.33** | 20.39 ± 0.04 | **-0.37** |

*Every architectural component yields a strictly positive contribution ($\Delta < 0$ when removed).*

---

### Modality Isolation (Visual-Only vs. Retrieval-Only Mode)

Evaluating the extreme mode where the visual pathway is completely zeroed out ($v_{\text{img}} = \mathbf{0}, g = 1.0$) demonstrates the rich semantic encoding of SemKB:

| Modality Setting | Gate State $g$ | Test Set Scope | BLEU-4 | METEOR | CIDEr (Mean ± SD) | SPICE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **VOB (Visual-Only Baseline)** |  = 0.00$ | 5,000 images | 33.68 ± 0.14 | 26.77 ± 0.10 | **107.15 ± 0.74** | 19.73 ± 0.10 |
| **Retrieval-Only (Old SemKB)** |  = 1.00$ | 5,000 images | 25.12 | 22.99 | **81.44** | 15.75 |
| **Retrieval-Only (Old SemKB)** |  = 1.00$ | 4,464 RelTR images | 25.52 ± 0.10 | 23.21 ± 0.03 | **81.86 ± 0.08** | 15.83 ± 0.08 |
| **Retrieval-Only (SemKB)** |  = 1.00$ | 5,000 images | 30.89 ± 0.02 | 26.06 ± 0.03 | **103.41 ± 0.16** *(+21.97 vs Old SemKB)* | 18.94 ± 0.02 |
| **Retrieval-Only (SemKB)** |  = 1.00$ | 4,464 RelTR images | 31.56 ± 0.01 | 26.39 ± 0.02 | **104.75 ± 0.13** *(+22.89 vs Old SemKB)* | 19.20 ± 0.01 |
| **SemRAG (Proposed Full Model)** |  \approx 0.50$ (learned) | 5,000 images | **34.90 ± 0.06** | **27.85 ± 0.02** | **115.41 ± 0.07** | **20.76 ± 0.02** |

---

### Parameter Sensitivity Analysis ($K$ and $M$)

* **Number of Retrieved Captions $K \in \{1, 3, 5, 10, 20\}$:**
  * $K=1 \rightarrow 112.13 \pm 0.04$ CIDEr (12.8 ms/img)
  * $K=3 \rightarrow 114.76 \pm 0.13$ CIDEr (13.0 ms/img)
  * **$K=5 \rightarrow \mathbf{115.41 \pm 0.07}$ CIDEr (16.8 ms/img) — Optimal Peak**
  * $K=10 \rightarrow 115.16 \pm 0.17$ CIDEr (24.1 ms/img)
  * $K=20 \rightarrow 114.81 \pm 0.09$ CIDEr (47.1 ms/img)
* **Number of Scene-Graph Triplets $M \in \{0, 3, 5, 10, 20\}$:**
  * $M=0 \rightarrow 112.08 \pm 0.33$ CIDEr
  * $M=3 \rightarrow 114.31 \pm 0.11$ CIDEr
  * $M=5 \rightarrow 115.26 \pm 0.04$ CIDEr
  * **$M=10 \rightarrow \mathbf{115.41 \pm 0.07}$ CIDEr — Optimal Peak**
  * $M=20 \rightarrow 115.18 \pm 0.02$ CIDEr

---

## 4. Interactive Web Application

An interactive web dashboard is provided in `UngDung/` for live demonstration and qualitative analysis.

* **Backend:** FastAPI, Uvicorn, PyTorch, FAISS, CLIP ViT-B/32, SafePTBTokenizer.
* **Frontend:** React 19, Vite, Tailwind CSS, Lucide Icons.
* **Key Features:**
  - Drag-and-drop support for `.png`, `.jpg`, `.jpeg`, and `.webp` images.
  - Side-by-side comparison mode: **Baseline Visual-Only ($g = 0$)** vs. **SemRAG with SemKB ($g \text{ adaptive})$**.
  - Dynamic gate value inspection and manual override slider.
  - Transparent visualization of retrieved candidate captions, similarity scores, and RelTR SPO relation triplets.
  - Detailed diagnostic panel detailing runtime latency and token probabilities.

---

## 5. Repository Structure & Notebook Index

```
NghienCuu_Nhom2/
├── .gitignore                                                      # Git exclusion rules
├── requirements.txt                                                # Unified Python dependencies
├── README.md                                                       # English documentation (this file)
├── README_VI.md                                                    # Vietnamese academic documentation
├── Mô hình kho tri thức mới.pdf                                    # Complete architectural diagram
├── readme_va_prompt_ngucanh_toanbo_duan_kb_v2_nhom02.md           # Full master context and audit log
│
├── Coco/                                                           # MS COCO 2014 Experiments (8 Notebooks)
│   ├── Knowledge_Base_V2_Improved_Full.ipynb                       # Build SemKB (566,747 entries)
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed42.ipynb
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed44.ipynb
│   ├── Baseline_Level_1_COCO2014_MultiSeed_BeamSearch_k3_Seed46.ipynb
│   ├── DanhGia_Level2_3Seeds_CaiTien_COCO_KetQua3.ipynb            # WF-RAG Evaluation (Seeds 42, 44, 46)
│   ├── DanhGia_Level3_3Seeds_CaiTien_COCO_KetQua3.ipynb            # RR-RAG Evaluation (Seeds 42, 44, 46)
│   ├── DanhGia_Level4_3Seeds_CaiTien_COCO_KetQua6.ipynb            # GM-RAG Evaluation (Seeds 42, 44, 46)
│   └── DanhGia_Level5_3Seeds_CaiTien_COCO_KetQua5.ipynb            # SemRAG Full Proposed Model (Seeds 42, 44, 46)
│
├── flick30k/                                                       # Flickr30k Experiments (11 Notebooks)
│   ├── PhuongAn_A/                                                 # In-Domain Flickr30k Evaluation
│   │   ├── Tao_Kho_Tri_Thuc_V2_Flickr30k_CaiTien_KetQua.ipynb     # Build Flickr30k SemKB (145k entries)
│   │   ├── DanhGia_Level1_3Seeds_Baseline_Flickr30k_KetQua.ipynb   # VOB Baseline Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level2_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # WF-RAG Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level3_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # RR-RAG Flickr30k (Seeds 42, 44, 46)
│   │   ├── DanhGia_Level4_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # GM-RAG Flickr30k (Seeds 42, 44, 46)
│   │   └── DanhGia_Level5_3Seeds_CaiTien_Flickr30k_KetQua.ipynb   # SemRAG Full Proposed Model (Seeds 42, 44, 46)
│   │
│   └── PhuongAn_B/                                                 # Zero-Shot Cross-Domain Transfer (COCO -> Flickr30k)
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level1_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level2_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level3_3Seeds_KetQua.ipynb
│       ├── PA_B_BeNguyenCOCO_ChamFlickr30k_Level4_3Seeds_KetQua.ipynb
│       └── PA_B_BeNguyenCOCO_ChamFlickr30k_Level5_3Seeds_KetQua.ipynb
│
├── Boc_Tach/                                                       # Ablation & Parameter Analyses (8 Notebooks)
│   ├── Chi_Dung_TriThuc/
│   │   └── DanhGia_Level6_ChiDungTriThuc_KB_V2_COCO_KetQua.ipynb   # Modality Isolation (Retrieval-Only Mode)
│   ├── Tach_Dimersity/
│   │   └── Ablation_01_Tach_MMR_Diversity_KB_V2_COCO_KetQua.ipynb # w/o MMR Diversity Selection
│   ├── Tach_Sematic/
│   │   └── Ablation_02_Tach_Semantic_Expansion_KB_V2_COCO_KetQua.ipynb # w/o WordNet & SPO Expansion
│   ├── Tach_ReRanking/
│   │   └── Ablation_03_Tach_RelTR_SPO_Reranking_KB_V2_COCO_KetQua.ipynb # w/o Scene-Graph RelTR Reranking
│   ├── Tach_MultiToken/
│   │   └── Ablation_04A_ChiTach_MultiToken_MeanPool_KB_V2_COCO_KetQua.ipynb # w/o Multi-Token Memory
│   ├── Tach_GateFusion/
│   │   └── Ablation_04B_ChiTach_GatedFusion_Linear_KB_V2_COCO_KetQua.ipynb  # w/o Dynamic Gated Fusion
│   ├── SPO/
│   │   └── KhaoSat_Bang8_SoBoBa_SPO_M_KB_V2_COCO_3Seeds_KetQua.ipynb # Sensitivity to Relation Triplets M
│   └── Thay đổi Top K/
│       └── KhaoSat_Bang9_ThayDoi_TopK_KB_V2_COCO_3Seeds_KetQua.ipynb # Sensitivity to Retrieved Captions K
│
└── UngDung/                                                        # Interactive Web Application
    ├── scripts/
    │   ├── download_level5.py                                      # Automatic asset downloader from Google Drive
    │   └── level5_drive.json                                       # Verified Drive file IDs & SHA-256 hashes
    ├── models/level5/                                              # Target directory for checkpoint & SemKB
    └── Fontend/
        ├── image-captioning-baseline/backend/                      # FastAPI Backend service
        └── webapp/                                                 # React 19 + Vite Frontend
```

---

## 6. Installation & Reproduction Guide

### Environment Setup

We recommend creating an isolated virtual environment using Python 3.10, 3.11, or 3.12:

```bash
# Clone the repository
git clone https://github.com/YourUsername/SemRAG-Captioning.git
cd SemRAG-Captioning

# Create and activate virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Downloading Model Checkpoints & Knowledge Base from Google Drive

Pretrained checkpoints and large SemKB embedding matrices exceed GitHub's 100MB file limit. We provide an automated downloader with SHA-256 integrity verification:

```bash
cd UngDung
python scripts/download_level5.py --manifest scripts/level5_drive.json
```

**Asset Manifest Reference (`level5_drive.json`):**
| Asset Name | Target Filename | Size | Drive File ID | SHA-256 Checksum |
| :--- | :--- | :--- | :--- | :--- |
| **Model Checkpoint** | `best_model.pt` | ~50 MB | `1SrRcYiCnzr3bnEs7aZv0xVMsbY4iFb9K` | `a708a1c41f98421d6af333a596d30d9c733034bf9ba57bf4484cce2c9357a9e4` |
| **Vocabulary** | `vocab.json` | ~300 KB | `1FrnEN43-COOQ3XYWu5kqfAZpTQqPWu22` | `c47a5de5309a21a22a5e9267505c8397781584c49eb29a626a3f8c595b6bdf83` |
| **CLIP Embeddings** | `text_embeddings_train_restval.npy`| 1.16 GB | `1-eriJH2wm82Fz-jEFrI0zMej521bSUAU` | `a7d701abb4efbd23ea548b8499785a4fc19a26070a0686d271e0ace17f9db225` |
| **Mapping Metadata**| `text_embedding_mapping.jsonl` | 266 MB | `1huUhetLNK80nqs5JOAd7tP9Wj9PetTAr` | `91f8a7dd2b70f88ec14c0ab6cc67ec9de2dd2b32062d903d0698801f53b7cf32` |

### Running Evaluation Notebooks

To reproduce any benchmark table from the thesis, open Jupyter Notebook:
```bash
jupyter notebook
```
Navigate to `Coco/`, `flick30k/`, or `Boc_Tach/` and run all cells. All 27 notebooks are pre-executed with zero errors, producing deterministic outputs matching the reported tables.

### Launching the Web Application

1. **Start the FastAPI Backend:**
   ```bash
   cd UngDung/Fontend/image-captioning-baseline
   uvicorn backend.main:app --host 127.0.0.1 --port 8000
   ```
   API Swagger documentation will be available at `http://127.0.0.1:8000/docs`.

2. **Start the React Frontend:**
   ```bash
   cd UngDung/Fontend/webapp
   npm install
   npm run dev
   ```
   Open your browser and navigate to `http://127.0.0.1:5173`.

---

## 7. Citation & BibTeX

If you find this research, code, or SemKB knowledge base useful in your academic work, please cite our graduation thesis:

```bibtex
@thesis{SemRAG_HUIT_2026,
  author    = {Ngo, Quoc Binh and Nguyen, Thi Quyen and Nguyen, Quoc Viet},
  title     = {Scene-Graph Guided Retrieval-Augmented Image Captioning with Gated Multi-Token Fusion and Structured Knowledge Base},
  school    = {Faculty of Information Technology, Ho Chi Minh City University of Industry and Trade (HUIT)},
  year      = {2026},
  month     = {November},
  address   = {Ho Chi Minh City, Vietnam},
  type      = {Undergraduate Thesis},
  note      = {Supervised by M.Sc. Dinh Thi Man and Ph.D. Nguyen Van Thinh}
}
```

---

## 8. License & Acknowledgments

* **Code License:** [MIT License](LICENSE).
* **Datasets & Pretrained Models:**
  - MS COCO Consortium ([MS COCO Terms of Use](https://cocodataset.org/))
  - Flickr30k Entity Dataset ([Flickr30k Terms](https://shannon.cs.illinois.edu/DenotationGraph/))
  - OpenAI CLIP ([MIT License](https://github.com/openai/CLIP/blob/main/LICENSE))
  - RelTR ([Apache 2.0 License](https://github.com/yrcong/RelTR))
* **Acknowledgments:** We express our sincere gratitude to **M.Sc. Dinh Thi Man** and **Ph.D. Nguyen Van Thinh** for their invaluable guidance, constructive insights, and mentorship throughout this research project at HUIT.

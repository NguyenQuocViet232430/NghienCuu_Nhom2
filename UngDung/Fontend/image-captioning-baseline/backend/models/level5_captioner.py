"""Seed-46 native five-token Level 5A inference.

Architecture/state keys follow the project's multiseed_5tokens_v1/level5_model.py:
CLIP 512 -> decoder 256, 3 layers, 8 heads; one scalar *knowledge* gate.
No randomly initialized or partially loaded model is ever served.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)
ARCHITECTURE_ID = "level5_native5_trained_v1"
MODEL_NAME = "SemRAG: Semantic Retrieval-Augmented Image Captioning (Kho SemKB V2)"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models" / "level5"
STOP_WORDS = {"a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "with", "and", "of", "to"}
ALIASES = {
    "airplane": ["plane", "aeroplane", "aircraft", "jet"], "bicycle": ["bike", "cycle"],
    "motorcycle": ["motorbike", "moto", "bike"], "couch": ["sofa"],
    "potted plant": ["plant", "houseplant", "flowerpot"],
    "cell phone": ["phone", "mobile phone", "smartphone"], "dining table": ["table", "dinner table"],
    "tv": ["television", "monitor", "screen"], "sports ball": ["ball"],
    "baseball glove": ["mitt", "baseball mitt"],
}


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_linear = nn.Linear(1025, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, v_img, k_tokens, scores):
        pooled = torch.bmm(F.softmax(scores, dim=-1).unsqueeze(1), k_tokens).squeeze(1)
        gate_input = torch.cat((F.normalize(v_img, dim=-1), F.normalize(pooled, dim=-1),
                                scores.mean(dim=-1, keepdim=True)), dim=-1)
        return self.sigmoid(self.gate_linear(gate_input))


class DualBranchDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
        self.cross_vis = nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
        self.cross_knw = nn.MultiheadAttention(256, 8, dropout=0.1, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(256, 1024), nn.Linear(1024, 256)
        self.activation = nn.ReLU()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(256) for _ in range(3))
        self.drop = nn.Dropout(0.1)

    def forward(self, tgt, m_vis, m_knw, gate, tgt_mask, tgt_pad_mask):
        h = self.norm1(tgt + self.drop(self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_pad_mask)[0]))
        visual = self.cross_vis(h, m_vis, m_vis)[0]
        knowledge = self.cross_knw(h, m_knw, m_knw)[0]
        g = gate.unsqueeze(1)
        h = self.norm2(h + self.drop((1 - g) * visual + g * knowledge))
        return self.norm3(h + self.drop(self.linear2(self.drop(self.activation(self.linear1(h))))))


class NativeFiveTokenModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.token_embedding = nn.Embedding(vocab_size, 256, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(30, 256)
        self.visual_proj = nn.Linear(512, 256)
        self.knowledge_proj = nn.Linear(512, 256)
        self.gated_fusion = GatedFusion()
        self.layers = nn.ModuleList([DualBranchDecoderLayer() for _ in range(3)])
        self.output_projection = nn.Linear(256, vocab_size)

    def forward(self, v_img, k_tokens, scores, input_ids, gate_override=None, gate_boost=0.0):
        batch, length = input_ids.shape
        if k_tokens.shape != (batch, 5, 512) or scores.shape != (batch, 5):
            raise ValueError("Level 5A requires exactly [B,5,512] caption embeddings and [B,5] scores")
        gate = self.gated_fusion(v_img, k_tokens, scores)
        if gate_override is not None:
            gate = torch.full_like(gate, gate_override)
        elif gate_boost > 0.0:
            gate = torch.clamp(gate + gate_boost, max=0.82)
        m_vis = self.visual_proj(v_img).unsqueeze(1)
        m_knw = self.knowledge_proj(k_tokens)
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        h = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        for layer in self.layers:
            h = layer(h, m_vis, m_knw, gate, mask, input_ids.eq(self.pad_id))
        return self.output_projection(h), gate


def load_checkpoint(path: Path, vocab_path: Path, device: str, expected_seed: int = 46):
    vocab = json.loads(vocab_path.read_text(encoding="utf-8-sig"))
    stoi = vocab.get("stoi", vocab)
    if (not isinstance(stoi, dict) or not all(isinstance(k, str) and type(v) is int for k, v in stoi.items())
            or set(stoi.values()) != set(range(len(stoi)))):
        raise ValueError("vocab.json must contain a contiguous, unique stoi mapping")
    for special in ("<PAD>", "<BOS>", "<EOS>", "<UNK>"):
        if special not in stoi:
            raise ValueError(f"Vocabulary missing {special}; use the training vocab.json")
    # Do not silently fall back to unrestricted pickle or strict=False.
    # The training contract stores torch.__version__ as this str subclass.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected best_model.pt export with model_state_dict, contract and evidence")
    contract, evidence = checkpoint.get("contract", {}), checkpoint.get("evidence", {})
    seeds = [value for value in (contract.get("seed"), evidence.get("training_seed"), checkpoint.get("seed")) if value is not None]
    if not seeds or any(int(seed) != expected_seed for seed in seeds):
        raise ValueError(f"Checkpoint seed is missing or is not Seed {expected_seed}: {seeds}")
    if evidence.get("trained") is not True or evidence.get("optimizer_steps", 0) <= 0:
        raise ValueError("Checkpoint lacks evidence of completed Level 5A training")
    if contract.get("architecture") != ARCHITECTURE_ID:
        raise ValueError(f"Unsupported checkpoint architecture: {contract.get('architecture')}")
    memory = contract.get("memory_contract", {})
    if memory.get("knowledge_memory_tokens") != 5 or memory.get("tokens_per_caption") != 1:
        raise ValueError("Checkpoint is not the native five-caption / five-token model")
    model = NativeFiveTokenModel(len(stoi), stoi["<PAD>"])
    state = checkpoint["model_state_dict"]
    if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Checkpoint has invalid or non-finite weights")
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    state_sha256 = digest.hexdigest()
    if evidence.get("model_state_sha256", state_sha256) != state_sha256:
        raise ValueError("Checkpoint tensor hash does not match its training evidence")
    model.requires_grad_(False).to(device).eval()
    return model, stoi, {"seed": expected_seed, "architecture": ARCHITECTURE_ID,
                         "state_sha256": state_sha256,
                         "d_model": 256, "num_layers": 3, "num_heads": 8,
                         "knowledge_memory_tokens": 5, "gate_semantics": "knowledge_weight",
                         "gate_scope": "one_scalar_per_image", "strict_load": True}


class KnowledgeBase:
    """Build IndexFlatIP from the same ordered embeddings consumed by the decoder."""
    def __init__(self, model_dir: Path):
        import faiss
        self.embeddings = np.load(model_dir / "text_embeddings_train_restval.npy", mmap_mode="r", allow_pickle=False)
        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 512 or len(self.embeddings) < 50:
            raise ValueError("Knowledge embeddings must be [N>=50,512]")
        self.captions, self.image_ids = [], []
        self.semantic_data = []
        with (model_dir / "text_embedding_mapping.jsonl").open(encoding="utf-8-sig") as stream:
            for i, line in enumerate(stream):
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("row_id", i)) != i or row.get("split") not in ("train", "restval"):
                    raise ValueError(f"KB row {i}: wrong row order or non-training split")
                if row.get("type", "caption") not in ("caption", "text"):
                    raise ValueError(f"KB row {i}: only caption/text records are supported")
                caption = row.get("caption") or row.get("raw_caption")
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"KB row {i}: missing caption")
                self.captions.append(caption.strip())
                self.image_ids.append(row.get("image_id"))

                raw_objs = row.get("objects", [])
                c_objs = []
                for o in raw_objs:
                    if isinstance(o, dict):
                        lbl = o.get("label") or o.get("name")
                        if lbl:
                            c_objs.append(str(lbl).lower())
                    elif isinstance(o, str) and o.strip():
                        c_objs.append(o.strip().lower())

                c_triples = []
                for r in row.get("relations", []):
                    if isinstance(r, dict):
                        s, p, obj = r.get("subject", ""), r.get("predicate", ""), r.get("object", "")
                        if s and p and obj:
                            c_triples.append(f"{s} {p} {obj}")
                    elif isinstance(r, (list, tuple)) and len(r) == 3:
                        c_triples.append(f"{r[0]} {r[1]} {r[2]}")
                for a in row.get("attributes", []):
                    if isinstance(a, dict):
                        s, val = a.get("subject", ""), a.get("value", "")
                        if s and val:
                            c_triples.append(f"{s} has {val}")
                for ac in row.get("actions", []):
                    if isinstance(ac, dict):
                        s, pred = ac.get("subject", ""), ac.get("predicate", "")
                        if s and pred:
                            c_triples.append(f"{s} {pred}")

                if c_objs or c_triples:
                    self.semantic_data.append({"objects": c_objs, "triples": c_triples})
                else:
                    self.semantic_data.append(None)

        if len(self.captions) != len(self.embeddings):
            raise ValueError("Knowledge mapping and embedding row counts differ")
        self.index = faiss.IndexFlatIP(512)
        for start in range(0, len(self.embeddings), 8192):
            block = np.array(self.embeddings[start:start + 8192], dtype=np.float32, copy=True)
            if not np.isfinite(block).all() or not np.allclose(np.linalg.norm(block, axis=1), 1, atol=1e-3):
                raise ValueError("Knowledge embeddings must be finite and L2-normalized, as in training")
            self.index.add(block)

    def retrieve(self, query, expanded_labels, clip_weight=0.6, mmr_weight=0.7):
        if not 0 <= clip_weight <= 1 or not 0 <= mmr_weight <= 1:
            raise ValueError("Retrieval weights must be between zero and one")
        clip_scores, rows = self.index.search(np.ascontiguousarray(query, dtype=np.float32), 50)
        rows, clip_scores = rows[0], clip_scores[0]
        if np.any(rows < 0):
            raise ValueError("FAISS returned missing candidate rows")
        captions = [self.captions[int(i)] for i in rows]
        jaccard = []
        for caption in captions:
            words = set(re.findall(r"[a-z]+", caption.lower())) - STOP_WORDS
            union = expanded_labels | words
            jaccard.append(len(expanded_labels & words) / len(union) if union else 0.0)
        # Same min/max normalization, MMR and final sorting as rerank_one in training.
        raw = clip_weight * clip_scores.astype(np.float64) + (1 - clip_weight) * np.asarray(jaccard) if expanded_labels else clip_scores.astype(np.float64)
        span = float(np.ptp(raw))
        relevance = np.ones_like(raw) if span < 1e-6 else (raw - raw.min()) / span
        embeddings = np.array(self.embeddings[rows], dtype=np.float32)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        pairwise = normalized @ normalized.T
        selected, remaining = [], list(range(len(rows)))
        for _ in range(5):
            best = max(remaining, key=lambda i: mmr_weight * relevance[i] - (1 - mmr_weight) * (float(pairwise[i, selected].max()) if selected else 0))
            selected.append(best)
            remaining.remove(best)
        selected.sort(key=lambda i: (-relevance[i], i))
        items = []
        for i in selected:
            row_idx = int(rows[i])
            sem = self.semantic_data[row_idx] if row_idx < len(self.semantic_data) else None
            cand_objs = sem["objects"] if sem else []
            cand_triples = sem["triples"] if sem else []
            words = set(re.findall(r"[a-z]+", captions[i].lower())) - STOP_WORDS
            matched = sorted(set(cand_objs) | (words & expanded_labels))
            items.append({
                "id": f"c{row_idx}",
                "content": captions[i],
                "source": "Kho Tri Thức Đa Tầng SemKB (MS COCO)",
                "score": float(relevance[i]),
                "metadata": {
                    "row_id": row_idx,
                    "image_id": self.image_ids[row_idx],
                    "clip_score": float(clip_scores[i]),
                    "faiss_rank": i + 1,
                    "jaccard_score": float(jaccard[i]),
                    "hybrid_score": float(raw[i]),
                    "objects": cand_objs[:6],
                    "triples": cand_triples[:4],
                    "matched_objects": matched[:6]
                }
            })
        return embeddings[selected], relevance[selected].astype(np.float32), items


class SemanticExpansion:
    def __init__(self, model_dir: Path | None = None):
        self.wordnet, self.error = None, None
        try:
            import nltk
            if model_dir is not None:
                local_nltk = Path(model_dir) / "nltk_data"
                if local_nltk.is_dir() and str(local_nltk) not in nltk.data.path:
                    nltk.data.path.insert(0, str(local_nltk))
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            self.wordnet = wordnet
        except (ImportError, LookupError) as exc:
            self.error = f"WordNet unavailable ({type(exc).__name__}); using detected labels and aliases only"
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            self.wordnet = wordnet
        except (ImportError, LookupError) as exc:
            self.error = f"WordNet unavailable ({type(exc).__name__}); using detected labels and aliases only"

    def expand(self, labels):
        expanded = set()
        for label in labels:
            clean = label.lower().strip().replace("_", " ")
            if not clean:
                continue
            # Stable insertion order avoids Python hash-seed dependent synonym truncation.
            values = dict.fromkeys([clean, *ALIASES.get(clean, [])])
            if self.wordnet is not None:
                for synset in self.wordnet.synsets(clean.replace(" ", "_"), pos=self.wordnet.NOUN):
                    for lemma in synset.lemmas():
                        word = lemma.name().lower().replace("_", " ")
                        if word.replace(" ", "").isalpha():
                            values.setdefault(word, None)
                        if len(values) >= 8:
                            break
                    if len(values) >= 8:
                        break
            expanded.update(list(values)[:8])
        return expanded


_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
CONSENSUS_STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "at", "to", "with", "and", "or", "for",
    "is", "are", "was", "were", "be", "been", "being", "by", "from", "as", "into",
    "this", "that", "these", "those", "it", "its", "their", "his", "her", "there"
}


def build_consensus_guidance(items, labels, stoi, device):
    """Khối dẫn dắt đồng thuận tri thức & SPO từ Cell 5 của Notebook đánh giá chuẩn COCO."""
    ref_captions = [it.get("content") or it.get("caption", "") for it in items]
    target_objs = [str(l).lower() for l in labels]
    spo_concept_words = set()
    for obj in target_objs:
        for w in _WORD_RE.findall(obj):
            if w not in CONSENSUS_STOPWORDS and len(w) > 1:
                spo_concept_words.add(w)

    word_doc_votes = collections.Counter()
    ngram_tf_profiles = []
    ref_lengths = []

    for idx, ref in enumerate(ref_captions):
        w_doc = 1.0 / (1.0 + 0.1 * idx)
        toks = _WORD_RE.findall(ref.lower())
        if not toks:
            continue
        ref_lengths.append(len(toks))
        for w in set(toks):
            if w not in CONSENSUS_STOPWORDS and len(w) > 2:
                word_doc_votes[w] += w_doc

        L = len(toks)
        ng_vecs = {1: {}, 2: {}, 3: {}, 4: {}}
        ng_norms = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        for n in (1, 2, 3, 4):
            if L >= n:
                cnts = collections.Counter(tuple(toks[k:k+n]) for k in range(L - n + 1))
                norm_sq = 0.0
                vec_n = {}
                for ng, c_val in cnts.items():
                    val = c_val / L
                    vec_n[ng] = val
                    norm_sq += val * val
                ng_vecs[n] = vec_n
                ng_norms[n] = math.sqrt(norm_sq) if norm_sq > 1e-12 else 0.0
        ngram_tf_profiles.append((w_doc, L, ng_vecs, ng_norms))

    bos, eos, pad, unk = (stoi.get(s) for s in ("<BOS>", "<EOS>", "<PAD>", "<UNK>"))
    lexical_bonus = torch.zeros(len(stoi), dtype=torch.float32, device=device)
    for w, vote_w in word_doc_votes.items():
        if w in stoi and stoi[w] not in (bos, eos, pad, unk):
            is_spo = w in spo_concept_words
            bonus = float(vote_w * (1.5 if is_spo else 1.0))
            lexical_bonus[stoi[w]] = min(1.2, bonus * 0.4)

    return {
        "ngram_tf_profiles": ngram_tf_profiles,
        "spo_concept_words": spo_concept_words,
        "lexical_bonus": lexical_bonus,
    }


def score_consensus(cand_str, guidance):
    """Chấm điểm CIDEr 1..4-gram + Độ phủ SPO từ Notebook Cell 5."""
    toks = _WORD_RE.findall(cand_str.lower())
    if not toks:
        return -1e9
    L_cand = len(toks)

    cand_vecs = {1: {}, 2: {}, 3: {}, 4: {}}
    cand_norms = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for n in (1, 2, 3, 4):
        if L_cand >= n:
            cnts = collections.Counter(tuple(toks[k:k+n]) for k in range(L_cand - n + 1))
            norm_sq = 0.0
            vec_n = {}
            for ng, cnt in cnts.items():
                val = cnt / L_cand
                vec_n[ng] = val
                norm_sq += val * val
            cand_vecs[n] = vec_n
            cand_norms[n] = math.sqrt(norm_sq) if norm_sq > 1e-12 else 0.0

    total_cider_sim = 0.0
    for w_doc, L_ref, ref_vecs, ref_norms in guidance["ngram_tf_profiles"]:
        len_diff = float(L_cand - L_ref)
        gauss_pen = math.exp(-(len_diff * len_diff) / (2.0 * 36.0))
        n_sim_sum = 0.0
        for n in (1, 2, 3, 4):
            c_norm = cand_norms[n]
            r_norm = ref_norms[n]
            if c_norm <= 0.0 or r_norm <= 0.0:
                continue
            dot = sum(c_val * ref_vecs[n][ng] for ng, c_val in cand_vecs[n].items() if ng in ref_vecs[n])
            n_sim_sum += dot / (c_norm * r_norm)
        total_cider_sim += w_doc * gauss_pen * (n_sim_sum / 4.0)

    spo_words = guidance["spo_concept_words"]
    tok_set = set(toks)
    spo_hits = sum(1 for sw in spo_words if sw in tok_set) if spo_words else 0
    spo_cov = min(1.0, spo_hits / max(2.0, min(5.0, float(len(spo_words))))) if spo_words else 0.0

    len_reg = -0.04 * abs(L_cand - 9.6)
    content_toks = [t for t in toks if t not in CONSENSUS_STOPWORDS and len(t) > 2]
    rep_pen = 0.25 * (len(content_toks) - len(set(content_toks)))

    return float(0.96 * (total_cider_sim * 3.5) + 0.42 * spo_cov + len_reg - rep_pen)


@torch.inference_mode()
def beam_search(model, visual, knowledge, scores, stoi, max_len=30, beam_size=3, alpha=0.6,
                gate_override=None, gate_boost=0.0, guidance=None):
    """Keep the best k live/finished beams; never re-expand EOS.
    max_len specifies maximum sequence length (up to 30 including BOS, or 1..29 generated tokens).
    When guidance is supplied, applies Cell 5 Candidate Harvesting (K=8) + Lexical Guidance + Consensus Re-scoring.
    """
    if max_len >= 30:
        steps = 29
    elif 1 <= max_len <= 29:
        steps = max_len
    else:
        raise ValueError("max_len must be between 1 and 30 for the trained positional embedding")
    bos, eos, pad = (stoi[s] for s in ("<BOS>", "<EOS>", "<PAD>"))
    itos = {i: word for word, i in stoi.items()}
    effective_beam = max(beam_size, 8) if guidance is not None else beam_size
    beams = [([bos], 0.0, False)]
    finished = []
    def rank(beam):
        return beam[1] / (((5 + max(1, len(beam[0]) - 1)) / 6) ** alpha)
    for _ in range(steps):
        candidates = []
        for tokens, score, done in beams:
            if done:
                candidates.append((tokens, score, True))
                continue
            inp = torch.tensor([tokens], dtype=torch.long, device=visual.device)
            try:
                logits, _ = model(visual, knowledge, scores, inp, gate_override=gate_override, gate_boost=gate_boost)
            except TypeError:
                logits, _ = model(visual, knowledge, scores, inp, gate_override=gate_override)
            next_logits = logits[0, -1].clone()
            if not torch.isfinite(next_logits).all():
                raise RuntimeError("Decoder returned non-finite logits")
            next_logits[[bos, pad]] = -torch.inf
            if not torch.isfinite(next_logits).any():
                raise RuntimeError("Decoder returned no finite token logits")

            if guidance is not None and len(tokens) >= 2:
                next_logits[tokens[-1]] -= 4.0

            lp = F.log_softmax(next_logits, dim=-1)
            if guidance is not None and "lexical_bonus" in guidance:
                max_lp = lp.max()
                plausible = (lp >= (max_lp - 4.0))
                lp = lp + plausible.float() * guidance["lexical_bonus"]

            probabilities, ids = lp.topk(min(effective_beam, len(stoi) - 2))
            for prob, idx in zip(probabilities, ids):
                is_done = (int(idx) == eos)
                new_t = tokens + [int(idx)]
                candidates.append((new_t, score + float(prob), is_done))
                if is_done and guidance is not None and len(new_t) >= 6:
                    finished.append((new_t, score + float(prob)))
        beams = sorted(candidates, key=rank, reverse=True)[:effective_beam]
        if all(done for _, _, done in beams):
            break

    if guidance is not None and finished:
        scored = []
        seen = set()
        for t, sc in finished:
            words = [itos[i] for i in t[1:] if i != eos]
            cap = " ".join(words).strip()
            if cap in seen or not cap:
                continue
            seen.add(cap)
            neural_sc = rank((t, sc))
            consensus_sc = score_consensus(cap, guidance)
            scored.append((neural_sc + consensus_sc, cap))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]

    tokens = max(beams, key=rank)[0][1:]
    words = []
    for token in tokens:
        if token == eos:
            break
        words.append(itos[token])
    return " ".join(words).strip()


class Level5Captioner:
    model_name = MODEL_NAME

    def __init__(self, model_dir=None, device=None):
        self.model_dir = Path(model_dir or os.environ.get("LEVEL5_MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
        self.device = device or os.environ.get("LEVEL5_DEVICE", "auto")
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.clip_model = self.kb = None
        self.reltr = None
        self.warnings = []
        self.clip_weight, self.mmr_weight = 0.6, 0.6
        self.gate_boost = float(os.environ.get("LEVEL5_GATE_BOOST", "0.13"))
        self.retrieval_config_source = "request_defaults"
        self._lock = threading.Lock()

    def load(self):
        for name in ("best_model.pt", "vocab.json", "text_embeddings_train_restval.npy", "text_embedding_mapping.jsonl"):
            if not (self.model_dir / name).is_file():
                raise FileNotFoundError(f"Missing {self.model_dir / name}. Run UngDung/scripts/download_level5.py first.")
        model, stoi, info = load_checkpoint(self.model_dir / "best_model.pt", self.model_dir / "vocab.json", self.device)
        config_path = self.model_dir / "results" / "final_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if (config.get("seed") != info["seed"] or config.get("architecture") != ARCHITECTURE_ID
                    or config.get("checkpoint_sha256") != info["state_sha256"]):
                raise ValueError("final_config.json does not belong to this checkpoint")
            self.clip_weight, self.mmr_weight = float(config["w"]), float(config["lambda_param"])
            if not 0 <= self.clip_weight <= 1 or not 0 <= self.mmr_weight <= 1:
                raise ValueError("Invalid retrieval weights in final_config.json")
            self.retrieval_config_source = "seed46_final_config"
        kb = KnowledgeBase(self.model_dir)
        if len(kb.captions) != 566747:
            raise ValueError("Seed 46 requires the complete 566747-row COCO train/restval knowledge base")
        import clip
        clip_model, preprocess = clip.load("ViT-B/32", device=self.device, jit=False,
                                          download_root=str(self.model_dir / "clip_cache"))
        clip_model.requires_grad_(False).eval()
        semantic = SemanticExpansion(self.model_dir)
        warnings = [semantic.error] if semantic.error else []
        try:
            from backend.models.reltr_adapter import RelTRDetector
            self.reltr = RelTRDetector.from_environment(self.device)
        except Exception as exc:
            if os.environ.get("LEVEL5_REQUIRE_RELTR", "0") == "1":
                raise RuntimeError(f"RelTR is required: {exc}") from exc
            warnings.append(f"RelTR unavailable: {exc}. Retrieval falls back to CLIP + MMR.")
            logger.warning(warnings[-1])
        self.stoi, self.model_info, self.semantic = stoi, info, semantic
        self.preprocess, self.clip_model, self.kb = preprocess, clip_model, kb
        self.warnings = warnings
        self.model = model  # Publish readiness only after every required asset loaded.

    def is_loaded(self):
        return self.model is not None and self.clip_model is not None and self.kb is not None

    def generate(self, image: Image.Image) -> str:
        return self.generate_details(image)["final_caption"]

    @torch.inference_mode()
    def generate_details(self, image: Image.Image):
        if not self.is_loaded():
            raise RuntimeError("Level 5A is not loaded")
        with self._lock:
            started = time.perf_counter()
            warnings = list(self.warnings)
            visual = self.clip_model.encode_image(self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)).float()
            visual = F.normalize(visual, dim=-1)
            labels, reltr_status = [], "unavailable"
            if self.reltr is not None:
                try:
                    labels = self.reltr.detect(image)
                    reltr_status = "active" if labels else "no_detections"
                except Exception as exc:
                    if os.environ.get("LEVEL5_REQUIRE_RELTR", "0") == "1":
                        raise
                    reltr_status = "inference_failed"
                    warnings.append(f"RelTR inference failed: {type(exc).__name__}; using CLIP + MMR")
                    logger.exception("RelTR inference failed")
            expanded = self.semantic.expand(labels)
            retrieval_start = time.perf_counter()
            embeddings, normalized_scores, items = self.kb.retrieve(visual.cpu().numpy(), expanded, self.clip_weight, self.mmr_weight)
            retrieval_time = time.perf_counter() - retrieval_start
            knowledge = torch.from_numpy(embeddings).unsqueeze(0).to(self.device)
            scores = torch.from_numpy(normalized_scores).unsqueeze(0).to(self.device)
            raw_gate = float(self.model.gated_fusion(visual, knowledge, scores).item())
            effective_gate = min(0.82, raw_gate + self.gate_boost) if self.gate_boost > 0.0 else raw_gate
            initial = beam_search(self.model, visual, knowledge, scores, self.stoi, gate_override=0.0)
            guidance = build_consensus_guidance(items, labels, self.stoi, self.device)
            final = beam_search(self.model, visual, knowledge, scores, self.stoi, gate_boost=self.gate_boost, guidance=guidance)
            if not final:
                raise RuntimeError("Decoder generated an empty caption")
            for item in items:
                item["metadata"].update(
                    gate_score=effective_gate,
                    raw_gate_score=raw_gate,
                    gate_boost=self.gate_boost,
                    gate_semantics="knowledge_weight",
                    gate_scope="image"
                )
            fallback = reltr_status != "active" or self.semantic.wordnet is None
            return {"initial_caption": initial, "final_caption": final,
                    "caption_keywords": list(dict.fromkeys(w for w in re.findall(r"[a-z]+", final.lower()) if w not in STOP_WORDS)),
                    "model_type": "rag_level5", "model_name": self.model_name, "device": self.device,
                    "processing_time": time.perf_counter() - started,
                    "knowledge_retrieval": True, "knowledge_ranking": True, "knowledge_fusion": True,
                    "retrieval": {"enabled": True, "top_k": 5, "items": items, "processing_time": retrieval_time},
                    "pipeline": {"architecture": self.model_info, "initial_caption_kind": "same_model_visual_only_ablation",
                                 "reltr_status": reltr_status, "wordnet_available": self.semantic.wordnet is not None,
                                 "semantic_reranking": bool(expanded), "fallback": fallback,
                                 "detected_labels": labels, "expanded_labels": sorted(expanded),
                                 "candidates": 50, "clip_weight": self.clip_weight, "mmr_weight": self.mmr_weight,
                                 "retrieval_config_source": self.retrieval_config_source,
                                 "beam_size": 3, "max_len": 30, "length_penalty": 0.6,
                                 "gate_boost": self.gate_boost, "raw_gate": raw_gate, "effective_gate": effective_gate,
                                 "kb_rows": len(self.kb.captions), "warnings": warnings}}

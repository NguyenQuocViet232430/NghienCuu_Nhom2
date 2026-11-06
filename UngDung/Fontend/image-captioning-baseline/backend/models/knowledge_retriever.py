"""
knowledge_retriever.py
======================
PHASE 2 — NOT YET IMPLEMENTED

This module defines the abstract interfaces for the Knowledge Retrieval pipeline.
These classes will be implemented in Phase 2 when the knowledge base and
FAISS index are ready.

Planned pipeline (Phase 2):
    Image
      ↓
    ImageEncoder  ──→  image_embedding
                              ↓
                    KnowledgeRetriever  ──→  top-K knowledge items
                              ↓
                    KnowledgeRanker  ──→  ranked knowledge
                              ↓
                    KnowledgeFusion  ──→  fused context vector
                              ↓
                    TransformerDecoder  ──→  enhanced caption

Phase 2 implementation steps:
    1. Build knowledge base (captions, triples, attributes from MS-COCO / Flickr30k).
    2. Encode knowledge items into embedding vectors.
    3. Build FAISS index over knowledge embeddings.
    4. Implement KnowledgeRetriever using FAISS nearest-neighbor search.
    5. Implement KnowledgeRanker (cosine similarity / cross-attention scoring).
    6. Implement KnowledgeFusion (concatenation / cross-attention / gating).
    7. Fine-tune / train TransformerDecoder with fused context.
    8. Evaluate on MS-COCO CIDEr / BLEU / METEOR vs. baseline.
"""

from abc import ABC, abstractmethod
from typing import Any


class KnowledgeRetriever(ABC):
    """Abstract interface for retrieving relevant knowledge items
    given an image embedding.

    Will use FAISS for approximate nearest-neighbor search over
    a pre-built knowledge embedding index.
    """

    @abstractmethod
    def retrieve(self, image_embedding: Any, top_k: int = 5) -> list:
        """Retrieve top-K knowledge items relevant to the image.

        Args:
            image_embedding: The encoded image vector.
            top_k: Number of knowledge items to retrieve.

        Returns:
            List of knowledge items (dicts with 'content', 'score', 'type').
        """
        raise NotImplementedError(
            "KnowledgeRetriever is not implemented yet. "
            "This is planned for Phase 2 — FAISS knowledge retrieval."
        )


class KnowledgeRanker(ABC):
    """Abstract interface for ranking and filtering retrieved knowledge items.

    Will apply cross-attention or cosine-similarity re-ranking to select
    the most contextually relevant subset from retrieved items.
    """

    @abstractmethod
    def rank(self, knowledge_items: list, image_embedding: Any = None) -> list:
        """Rank and filter knowledge items by relevance.

        Args:
            knowledge_items: Raw list of items from KnowledgeRetriever.
            image_embedding: Optional image vector for re-ranking.

        Returns:
            Sorted and filtered list of knowledge items.
        """
        raise NotImplementedError(
            "KnowledgeRanker is not implemented yet. "
            "This is planned for Phase 2 — knowledge ranking / filtering."
        )


class KnowledgeFusion(ABC):
    """Abstract interface for fusing image embedding with ranked knowledge.

    Will combine the image representation with retrieved knowledge context
    before feeding into the Transformer Decoder (caption generator).

    Fusion strategies (to be evaluated):
      - Simple concatenation
      - Cross-attention fusion
      - Gated fusion
    """

    @abstractmethod
    def fuse(self, image_embedding: Any, knowledge: list) -> Any:
        """Fuse image embedding with knowledge context.

        Args:
            image_embedding: Encoded image vector.
            knowledge: Ranked list of knowledge items.

        Returns:
            Fused context representation for the caption decoder.
        """
        raise NotImplementedError(
            "KnowledgeFusion is not implemented yet. "
            "This is planned for Phase 2 — knowledge-enhanced captioning."
        )

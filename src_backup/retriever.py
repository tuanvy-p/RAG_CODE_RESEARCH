from typing import List, Dict, Tuple, Optional, Any, Set
import re
import os
import time
import numpy as np
import faiss
import voyageai

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .ast_parser import CodeChunk
from .dependency_graph import DependencyGraph
from .config import config


class DenseRetriever:
    """
    Semantic Vector Retriever powered by Voyage AI Code Embedding API (voyage-code-4)
    and FAISS IndexFlatIP (Cosine Similarity).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY") or getattr(config.model, 'voyage_api_key', None)
        self.model_name = model_name or getattr(config.model, 'embedding_model', None) or "voyage-code-4"
        
        if not self.api_key:
            print("[Warning] VOYAGE_API_KEY is not set. Dense retrieval will not work until key is provided.")
            self.vo = None
        else:
            self.vo = voyageai.Client(api_key=self.api_key)

        # voyage-code-4 sử dụng default dimension là 1024
        self.dimension = 1024  
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product on L2-normalized vectors = Cosine Sim
        self.chunk_ids: List[str] = []
        self.chunks_map: Dict[str, CodeChunk] = {}

    def is_ready(self) -> bool:
        return self.vo is not None and self.index.ntotal > 0

    def embed_texts(self, texts: List[str], input_type: str = "document", batch_size: int = 32) -> np.ndarray:
        """
        Generates vector embeddings for a list of texts in batches using Voyage AI API.
        """
        if not self.vo:
            raise ValueError("Voyage API key is not configured. Please set VOYAGE_API_KEY in .env")

        if not texts:
            return np.empty((0, self.dimension), dtype="float32")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                result = self.vo.embed(
                    texts=batch,
                    model=self.model_name,
                    input_type=input_type
                )
                all_embeddings.extend(result.embeddings)
                time.sleep(0.5)  # Tránh tràn RPM tạm thời
            except Exception as e:
                print(f"[Error] Failed to embed batch {i}..{i+len(batch)} via Voyage AI API: {e}")
                # Fallback: fill with zero vectors if API fails for a batch
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
                time.sleep(5)

        vecs = np.array(all_embeddings, dtype="float32")
        # Update dimension động nếu Voyage trả về kích thước khác
        if vecs.shape[1] != self.dimension:
            self.dimension = vecs.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

        # Normalize L2 so Inner Product equals Cosine Similarity
        faiss.normalize_L2(vecs)
        return vecs

    def build_index(self, chunks: List[CodeChunk]):
        """
        Builds FAISS index from CodeChunk objects using their rich semantic representation via Voyage AI.
        """
        if not chunks:
            return

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunks_map = {c.chunk_id: c for c in chunks}
        
        # Format each chunk into an information-rich text representation
        texts = [c.get_embedding_text() for c in chunks]
        
        print(f"[DenseRetriever] Generating semantic embeddings for {len(texts)} chunks via Voyage AI ({self.model_name})...")
        vectors = self.embed_texts(texts, input_type="document")
        
        # Reset and populate FAISS index
        self.index.reset()
        self.index.add(vectors)
        print(f"[DenseRetriever] Successfully indexed {self.index.ntotal} vectors in FAISS.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches FAISS index for top_k most semantically similar code chunks using Voyage query embedding.
        Returns: List of (chunk_id, cosine_score)
        """
        if not self.is_ready():
            return []

        query_vec = self.embed_texts([query], input_type="query")
        scores, indices = self.index.search(query_vec, top_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunk_ids):
                results.append((self.chunk_ids[idx], float(score)))

        return results


class BM25Retriever:
    """
    Sparse Lexical Retriever using BM25Okapi for exact keyword and identifier matching.
    """

    def __init__(self):
        self.chunks_map: Dict[str, CodeChunk] = {}
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None

    def build_index(self, chunks: List[CodeChunk]):
        """
        Tokenizes chunks and builds BM25 index.
        """
        if not chunks:
            return

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunks_map = {c.chunk_id: c for c in chunks}
        
        # Tokenize code including file path, name, signature, and body
        self.corpus_tokens = [
            self._tokenize(f"{c.file_path} {c.name} {c.signature or ''} {c.code}")
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print(f"[BM25Retriever] Successfully indexed {len(self.chunk_ids)} chunks for lexical search.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Searches BM25 index for keyword/symbol matches.
        Returns: List of (chunk_id, bm25_score)
        """
        if not self.bm25 or not self.chunk_ids:
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:  # Only return chunks with actual token overlap
                results.append((self.chunk_ids[idx], score))

        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Smart code tokenizer: splits identifiers on camelCase, snake_case, and non-alphanumeric chars.
        """
        raw_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
        split_tokens = []
        for tok in raw_tokens:
            split_tokens.append(tok.lower())
            sub_tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', tok)
            for sub in sub_tokens:
                split_tokens.append(sub.lower())
        return split_tokens


class HybridRetriever:
    """
    Hybrid Retriever that unifies:
    1. Dense Retrieval (Voyage AI Vector Embeddings + FAISS)
    2. Sparse Retrieval (BM25 Lexical Matching)
    3. Graph-based Context Expansion (DependencyGraph 1-hop / 2-hop traversal)
    4. Rank Fusion via Reciprocal Rank Fusion (RRF) & Score Weighting
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        dependency_graph: Optional[DependencyGraph] = None,
        rrf_constant: int = 60
    ):
        self.dense = dense_retriever or DenseRetriever()
        self.bm25 = bm25_retriever or BM25Retriever()
        self.graph = dependency_graph or DependencyGraph()
        self.rrf_k = rrf_constant
        self.all_chunks_map: Dict[str, CodeChunk] = {}

    def index_repository(self, chunks: List[CodeChunk]):
        """
        Indexes the entire repository across all 3 subsystems simultaneously:
        - Semantic Dense Embeddings (FAISS)
        - Sparse BM25
        - Dependency Graph (NetworkX)
        """
        self.all_chunks_map = {c.chunk_id: c for c in chunks}

        # 1. Build Dependency Graph
        print("[HybridRetriever] Step 1/3: Building Dependency Graph...")
        self.graph.build_from_chunks(chunks)

        # 2. Build BM25 Index
        print("[HybridRetriever] Step 2/3: Building BM25 Index...")
        self.bm25.build_index(chunks)

        # 3. Build Dense Vector Index
        print("[HybridRetriever] Step 3/3: Building Dense Vector Index...")
        try:
            self.dense.build_index(chunks)
        except Exception as e:
            print(f"[Warning] Dense indexing skipped or failed: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        expand_graph: bool = True,
        hops: int = 1
    ) -> List[CodeChunk]:
        """
        Performs Hybrid Search & Context Expansion for a given query.
        Returns the top_k most relevant CodeChunk objects.
        """
        candidate_pool_size = max(top_k * 2, 10)

        # 1. Dense Semantic Search
        dense_results: List[Tuple[str, float]] = []
        if self.dense.is_ready():
            dense_results = self.dense.search(query, top_k=candidate_pool_size)

        # 2. Sparse BM25 Search
        sparse_results = self.bm25.search(query, top_k=candidate_pool_size)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}

        # Add Dense ranks
        for rank, (chunk_id, _) in enumerate(dense_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (dense_weight / (self.rrf_k + rank + 1))

        # Add Sparse ranks
        for rank, (chunk_id, _) in enumerate(sparse_results):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (sparse_weight / (self.rrf_k + rank + 1))

        # If both dense and sparse returned nothing, return empty
        if not rrf_scores:
            return []

        # Sort seed chunks by combined RRF score
        sorted_seeds = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        seed_ids = [item[0] for item in sorted_seeds[:top_k]]

        # 4. Graph Context Expansion (1-hop / 2-hop neighbors)
        final_chunk_ids = list(seed_ids)
        if expand_graph and self.graph.graph.number_of_nodes() > 0:
            expanded_ids = self.graph.expand_context_for_seeds(
                seed_chunk_ids=seed_ids,
                hops=hops,
                max_expanded=max(2, top_k // 2)
            )
            for eid in expanded_ids:
                if eid not in final_chunk_ids:
                    final_chunk_ids.append(eid)

        # Truncate to top_k and map back to CodeChunk objects
        final_chunks: List[CodeChunk] = []
        for cid in final_chunk_ids[:top_k]:
            if cid in self.all_chunks_map:
                final_chunks.append(self.all_chunks_map[cid])

        return final_chunks
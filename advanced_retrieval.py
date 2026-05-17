"""
Advanced Retrieval Module
=========================

This module implements multiple retrieval strategies:
1. Dense retrieval (semantic/vector search)
2. Hybrid retrieval (dense + keyword/BM25)
3. Re-ranking (improve result quality)
4. Multi-query retrieval (query expansion)

For production-grade RAG systems.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
import re
from collections import Counter


@dataclass
class RetrievalResult:
    """
    Represents a retrieved chunk with all its information.
    
    Attributes:
        content (str): The actual text content
        metadata (dict): Metadata (source, heading, etc.)
        score (float): Relevance score (0-1, higher is better)
        retrieval_method (str): How this was retrieved
        rank (int): Position in results (1 = best)
    """
    content: str
    metadata: Dict
    score: float
    retrieval_method: str
    rank: int
    
    def __repr__(self):
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return (f"RetrievalResult(rank={self.rank}, score={self.score:.3f}, "
                f"method={self.retrieval_method}, preview='{preview}')")


class BM25Retriever:
    """
    BM25 (Best Match 25) - Keyword-based retrieval.
    
    This is the classic "keyword search" algorithm used by search engines.
    It finds documents that contain the query keywords, with smart weighting.
    
    Good for:
    - Exact term matching (e.g., "Form 1764")
    - Technical terms
    - Proper nouns (e.g., "Texas DMV")
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Length normalization parameter (default 0.75)
        
        Note:
            These are the standard BM25 parameters used by Elasticsearch.
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.term_frequencies = []
        self.document_frequencies = {}
        self.N = 0  # Total number of documents
        
        print(f"✅ BM25 Retriever initialized (k1={k1}, b={b})")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Convert text into tokens (words).
        
        Args:
            text (str): Input text
            
        Returns:
            List of lowercase tokens
        
        Example:
            >>> tokenize("What are the Requirements?")
            ['what', 'are', 'the', 'requirements']
        """
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def index(self, documents: List[str]):
        """
        Index documents for BM25 search.
        
        This pre-processes all documents to enable fast searching.
        
        Args:
            documents (list): List of document texts
        
        What it does:
            1. Tokenizes each document
            2. Calculates term frequencies
            3. Builds document frequency index
            4. Computes average document length
        """
        print(f"\n🔍 Indexing {len(documents)} documents for BM25...")
        
        self.documents = documents
        self.N = len(documents)
        self.doc_lengths = []
        self.term_frequencies = []
        all_terms = set()
        
        # Step 1: Tokenize all documents and count term frequencies
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            # Count how often each term appears in this document
            tf = Counter(tokens)
            self.term_frequencies.append(tf)
            all_terms.update(tokens)
        
        # Step 2: Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Step 3: Calculate document frequencies (how many docs contain each term)
        self.document_frequencies = {}
        for term in all_terms:
            # Count in how many documents this term appears
            df = sum(1 for tf in self.term_frequencies if term in tf)
            self.document_frequencies[term] = df
        
        print(f"   ✅ Indexed {len(all_terms)} unique terms")
        print(f"   Average document length: {self.avg_doc_length:.1f} tokens")
    
    def _calculate_idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency (IDF) for a term.
        
        IDF measures how "rare" or "special" a term is:
        - Common words (the, is, and) → low IDF
        - Rare words (surety, eLICENSING) → high IDF
        
        Formula: IDF = log((N - df + 0.5) / (df + 0.5))
        """
        df = self.document_frequencies.get(term, 0)
        idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        return idf
    
    def _calculate_score(self, query_terms: List[str], doc_idx: int) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query tokens
            doc_idx: Index of the document to score
            
        Returns:
            BM25 score (higher = more relevant)
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_idx]
        tf_dict = self.term_frequencies[doc_idx]
        
        for term in query_terms:
            if term not in tf_dict:
                continue
            
            # Term frequency in this document
            tf = tf_dict[term]
            
            # Inverse document frequency
            idf = self._calculate_idf(term)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            
        Returns:
            List of (document_index, score) tuples, sorted by score
        
        Example:
            >>> results = bm25.search("surety bond requirements", top_k=5)
            >>> for idx, score in results:
            ...     print(f"Doc {idx}: score={score:.3f}")
        """
        # Tokenize query
        query_terms = self._tokenize(query)
        
        # Score all documents
        scores = []
        for doc_idx in range(self.N):
            score = self._calculate_score(query_terms, doc_idx)
            scores.append((doc_idx, score))
        
        # Sort by score (highest first) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class ReRanker:
    """
    Re-Ranker using Cross-Encoder model.
    
    What it does:
        Takes initial retrieval results and re-scores them using a more
        sophisticated model that directly compares query and document.
    
    Why it's better:
        - Dense retrieval: Fast but approximate
        - Re-ranking: Slower but very accurate
        
    Strategy:
        1. Use dense retrieval to get top 50 candidates (fast)
        2. Use re-ranker on those 50 to pick best 10 (accurate)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the re-ranker.
        
        Args:
            model_name: HuggingFace cross-encoder model
        
        Default model:
            ms-marco-MiniLM-L-6-v2 - Fast and effective for most tasks
        """
        print(f"\n🔄 Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("   ✅ Re-ranker loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents for a given query.
        
        Args:
            query (str): The search query
            documents (list): List of document texts to re-rank
            top_k (int): Number of results to return
            
        Returns:
            List of (original_index, score) tuples, sorted by relevance
        
        How it works:
            The cross-encoder looks at query + document together and
            predicts how relevant the document is to the query.
            
        Example:
            >>> initial_results = [doc1, doc2, doc3, ...]
            >>> reranked = reranker.rerank("bond requirements", initial_results, top_k=5)
            >>> # Now reranked[0] is the BEST match
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Create (index, score) tuples
        results = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


class HybridRetriever:
    """
    Hybrid Retrieval: Combines dense (semantic) and sparse (keyword) search.
    
    Why hybrid?
        - Dense search: Good for semantic similarity ("bond" matches "surety")
        - Sparse search: Good for exact terms ("Form 1764" must match exactly)
        - Together: Best of both worlds!
    
    Example:
        Query: "What is form 1764 about?"
        
        Dense only: Might miss the exact form number
        Sparse only: Might miss semantically similar content
        Hybrid: Finds documents with form number AND relevant content
    """
    
    def __init__(
        self,
        collection,
        embedding_model: SentenceTransformer,
        use_reranking: bool = True,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            collection: ChromaDB collection
            embedding_model: Sentence transformer model for dense retrieval
            use_reranking: Whether to use re-ranking (recommended)
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
        
        Note:
            dense_weight + sparse_weight should = 1.0
        """
        self.collection = collection
        self.embedding_model = embedding_model
        self.use_reranking = use_reranking
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize BM25 for sparse retrieval
        self.bm25 = BM25Retriever()
        
        # Initialize re-ranker if enabled
        self.reranker = None
        if use_reranking:
            self.reranker = ReRanker()
        
        print(f"\n🔄 Hybrid Retriever initialized:")
        print(f"   Dense weight: {dense_weight}")
        print(f"   Sparse weight: {sparse_weight}")
        print(f"   Re-ranking: {'enabled' if use_reranking else 'disabled'}")
    
    def index_for_sparse(self):
        """
        Index all documents in ChromaDB for BM25 (sparse) retrieval.
        
        This should be called once after adding documents to the collection.
        """
        print("\n📚 Indexing documents for sparse retrieval...")
        
        # Get all documents from collection
        results = self.collection.get()
        documents = results['documents']
        
        if not documents:
            print("   ⚠️  No documents in collection to index!")
            return
        
        # Index for BM25
        self.bm25.index(documents)
        
        print(f"   ✅ Indexed {len(documents)} documents")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid approach.
        
        Process:
            1. Dense retrieval (semantic search via embeddings)
            2. Sparse retrieval (keyword search via BM25)
            3. Combine scores with weights
            4. Re-rank top candidates (if enabled)
            5. Return final results
        
        Args:
            query (str): Search query
            top_k (int): Number of results to return
            where (dict): Metadata filters for ChromaDB
            where_document (dict): Document content filters
            
        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        print(f"\n🔍 Hybrid retrieval for: '{query}'")
        
        # Step 1: Dense retrieval (semantic search)
        print("   1️⃣  Dense retrieval (semantic)...")
        
        # Get more candidates than needed for re-ranking
        initial_k = top_k * 5 if self.use_reranking else top_k
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        dense_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(initial_k, self.collection.count()),
            where=where,
            where_document=where_document,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Convert distances to similarity scores (0-1)
        dense_docs = dense_results['documents'][0]
        dense_metadata = dense_results['metadatas'][0]
        dense_distances = dense_results['distances'][0]
        dense_scores = [1 - d for d in dense_distances]  # Convert distance to similarity
        
        print(f"      Retrieved {len(dense_docs)} candidates")
        print(f"      Score range: {min(dense_scores):.3f} - {max(dense_scores):.3f}")
        
        # Step 2: Sparse retrieval (keyword search)
        print("   2️⃣  Sparse retrieval (BM25)...")
        
        bm25_results = self.bm25.search(query, top_k=initial_k)
        
        # Normalize BM25 scores to 0-1 range
        if bm25_results:
            max_bm25_score = max(score for _, score in bm25_results)
            if max_bm25_score > 0:
                bm25_scores_norm = {idx: score / max_bm25_score 
                                   for idx, score in bm25_results}
            else:
                bm25_scores_norm = {idx: 0 for idx, _ in bm25_results}
        else:
            bm25_scores_norm = {}
        
        print(f"      Retrieved {len(bm25_results)} candidates")
        
        # Step 3: Combine scores
        print("   3️⃣  Combining dense + sparse scores...")
        
        # Create a mapping of document content to index for BM25 results
        doc_to_idx = {doc: i for i, doc in enumerate(dense_docs)}
        
        combined_results = []
        for i, (doc, metadata, dense_score) in enumerate(zip(dense_docs, dense_metadata, dense_scores)):
            # Get BM25 score if available
            bm25_score = bm25_scores_norm.get(i, 0)
            
            # Combine scores
            final_score = (self.dense_weight * dense_score + 
                          self.sparse_weight * bm25_score)
            
            combined_results.append({
                'content': doc,
                'metadata': metadata,
                'score': final_score,
                'dense_score': dense_score,
                'sparse_score': bm25_score
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top candidates for re-ranking
        candidates = combined_results[:initial_k]
        
        print(f"      Combined {len(candidates)} candidates")
        print(f"      Score range: {min(c['score'] for c in candidates):.3f} - "
              f"{max(c['score'] for c in candidates):.3f}")
        
        # Step 4: Re-ranking (if enabled)
        if self.use_reranking and self.reranker and len(candidates) > top_k:
            print("   4️⃣  Re-ranking with cross-encoder...")
            
            candidate_docs = [c['content'] for c in candidates]
            reranked_indices = self.reranker.rerank(query, candidate_docs, top_k=top_k)
            
            # Reorder results based on re-ranking
            final_results = []
            for rank, (idx, rerank_score) in enumerate(reranked_indices, 1):
                result = candidates[idx]
                final_results.append(RetrievalResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    score=rerank_score,
                    retrieval_method='hybrid+rerank',
                    rank=rank
                ))
            
            print(f"      Re-ranked to top {len(final_results)} results")
        else:
            # No re-ranking, just use combined results
            final_results = []
            for rank, result in enumerate(candidates[:top_k], 1):
                final_results.append(RetrievalResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    score=result['score'],
                    retrieval_method='hybrid',
                    rank=rank
                ))
        
        print(f"\n   ✅ Final results: {len(final_results)}")
        if final_results:
            print(f"      Best score: {final_results[0].score:.3f}")
            print(f"      Worst score: {final_results[-1].score:.3f}")
        
        return final_results


def example_usage():
    """Example of how to use the advanced retrieval system."""
    
    print("="*80)
    print("ADVANCED RETRIEVAL - EXAMPLE")
    print("="*80)
    
    # This example assumes you have a ChromaDB collection set up
    # See the complete RAG application for full integration
    
    print("\n📚 This module provides:")
    print("   1. BM25 (keyword-based retrieval)")
    print("   2. Re-Ranker (improves result quality)")
    print("   3. Hybrid Retriever (combines dense + sparse + reranking)")
    
    print("\n💡 See complete_rag_app.py for full integration!")


if __name__ == "__main__":
    example_usage()

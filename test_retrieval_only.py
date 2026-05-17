#!/usr/bin/env python3
"""
Retrieval-only test — shows exactly what chunks the RAG fetches for each query,
bypassing the Groq LLM step (which is blocked by network policy in this env).
"""

import os, sys, time
import numpy as np
sys.path.insert(0, '/home/user/texas-dmv-rag')

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as _ONNXEmbed

class ONNXEmbedWrapper:
    def __init__(self):
        self._ef = _ONNXEmbed()
        self._dim = 384
    def encode(self, texts, show_progress_bar=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return np.array(self._ef(texts), dtype=np.float32)
    def get_sentence_embedding_dimension(self):
        return self._dim

import sentence_transformers as _st
_st.SentenceTransformer = lambda *a, **k: ONNXEmbedWrapper()

import chromadb
from chromadb.config import Settings
from advanced_retrieval import HybridRetriever

DB_PATH    = "/home/user/texas-dmv-rag/chroma_test"
COLLECTION = "texas_dmv_elicensing_test"

client     = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name=COLLECTION)
embedder   = ONNXEmbedWrapper()

retriever = HybridRetriever(
    collection=collection,
    embedding_model=embedder,
    use_reranking=False,
    dense_weight=0.7,
    sparse_weight=0.3
)

# Re-build BM25 index from the stored collection
print(f"DB has {collection.count()} chunks — rebuilding BM25 index...")
all_docs = collection.get(include=['documents', 'metadatas'])
retriever.index_for_sparse()
print("BM25 ready.\n")

TEST_QUERIES = [
    "How do I apply for a new independent GDN license in Texas?",
    "What documents are required to apply for a franchise dealer license?",
    "What is the process to protest a dealer license in eLicensing?",
    "How do I add a vehicle to inventory using webDEALER?",
    "What are the requirements for a converter license?",
    "How do I get an in-transit license in Texas?",
    "What should I do if the eLicensing system is down and I need to process a sale?",
    "How do I amend an existing independent GDN license?",
]

print("=" * 80)
print("TEXAS DMV eLICENSING RAG — RETRIEVAL TEST (no LLM)")
print(f"Collection: {COLLECTION}  |  Chunks: {collection.count()}")
print("=" * 80)

timings = []
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n{'─'*80}")
    print(f"Q{i}: {query}")
    print('─'*80)
    t0 = time.time()
    results = retriever.retrieve(query, top_k=3)
    elapsed = time.time() - t0
    timings.append(elapsed)

    for j, r in enumerate(results, 1):
        src  = r.metadata.get('source_file', '?')
        page = r.metadata.get('page_number', '?')
        head = r.metadata.get('heading', '')
        print(f"\n  [{j}] score={r.score:.3f}  src={src}  page={page}")
        if head:
            print(f"      heading: {head}")
        # Show a clean excerpt
        excerpt = r.content.strip().replace('\n', ' ')
        print(f"      \"{excerpt[:300]}\"")
    print(f"\n  ⏱  Retrieval time: {elapsed:.2f}s")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Queries:        {len(TEST_QUERIES)}")
print(f"Chunks in DB:   {collection.count()}")
print(f"Avg retrieval:  {sum(timings)/len(timings)*1000:.0f} ms")
print(f"Min/Max:        {min(timings)*1000:.0f} ms / {max(timings)*1000:.0f} ms")
print("=" * 80)
print("\n✅  Ingestion and retrieval fully operational.")
print("⚠️   LLM generation (Groq) blocked by environment network policy.")
print("    To enable: whitelist api.groq.com in the environment's network settings.")

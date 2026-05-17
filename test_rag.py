#!/usr/bin/env python3
"""
Test script: ingest Google Drive PDFs into the Texas DMV RAG and run representative queries.
Uses ChromaDB's built-in ONNX MiniLM embedder (no HuggingFace download needed).
"""

import os, sys, time
import numpy as np
sys.path.insert(0, '/home/user/texas-dmv-rag')
if not os.environ.get('GROQ_API_KEY'):
    raise SystemExit("ERROR: GROQ_API_KEY env var not set. Run: export GROQ_API_KEY='your-key'")

# ── Wrap ONNX embedder to match SentenceTransformer's encode() interface ──────
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as _ONNXEmbed

class ONNXEmbedWrapper:
    """Drop-in replacement for SentenceTransformer using ChromaDB's bundled ONNX model."""
    def __init__(self):
        self._ef = _ONNXEmbed()
        self._dim = 384

    def encode(self, texts, show_progress_bar=False, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._ef(texts)
        return np.array(vecs, dtype=np.float32)

    def get_sentence_embedding_dimension(self):
        return self._dim

# Monkey-patch SentenceTransformer before any module imports it
import sentence_transformers as _st
_real_ST = _st.SentenceTransformer

def _patched_ST(model_name_or_path=None, *args, **kwargs):
    print(f"   [patch] Using local ONNX MiniLM instead of downloading '{model_name_or_path}'")
    return ONNXEmbedWrapper()

_st.SentenceTransformer = _patched_ST

# Also patch at module level so imports in sub-modules pick it up
import sentence_transformers
sentence_transformers.SentenceTransformer = _patched_ST

from fast_rag_universal import FastRAGSystemUniversal

DOCS_DIR = "/home/user/texas-dmv-rag/test_docs"
COLLECTION = "texas_dmv_elicensing_test"
DB_PATH    = "/home/user/texas-dmv-rag/chroma_test"

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

print("\n" + "=" * 80)
print("TEXAS DMV eLICENSING RAG — END-TO-END TEST")
print("=" * 80)

rag = FastRAGSystemUniversal(
    groq_api_key=os.environ['GROQ_API_KEY'],
    collection_name=COLLECTION,
    persist_directory=DB_PATH,
    chunk_size=700,
    chunk_overlap=120,
    use_hybrid_search=True,
    enable_profiling=True
)

# ── Ingestion ─────────────────────────────────────────────────────────────────
if rag.collection.count() == 0:
    print(f"\n📁 Ingesting documents from {DOCS_DIR}...")
    t_ingest = time.time()
    result = rag.add_folder(DOCS_DIR, recursive=False)
    ingest_elapsed = time.time() - t_ingest
    print(f"\n✅ Ingested: {result['files_processed']} files | "
          f"{result['total_chunks']} chunks | {ingest_elapsed:.1f}s")
else:
    print(f"\n✅ Collection already populated: {rag.collection.count()} chunks — skipping ingestion")

chunk_count = rag.collection.count()
print(f"   Total chunks in DB: {chunk_count}")

# ── Queries ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING TEST QUERIES")
print("=" * 80)

results = []
for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\n{'─'*80}")
    print(f"Q{i}: {query}")
    print('─'*80)
    t0 = time.time()
    try:
        resp = rag.query(query, top_k=5)
        elapsed = time.time() - t0
        answer  = resp.get('answer', resp.get('response', str(resp)))
        sources = resp.get('sources', [])
        src_names = list({s.get('source_file', s.get('file', '?')) for s in sources[:3]})
        print(f"ANSWER:\n{answer[:1000]}")
        print(f"\nSOURCES: {', '.join(src_names) if src_names else 'none'}")
        print(f"TIME: {elapsed:.1f}s")
        results.append({'query': query, 'answer': answer, 'time': elapsed, 'sources': src_names, 'ok': True})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR: {e}")
        results.append({'query': query, 'answer': '', 'time': elapsed, 'sources': [], 'ok': False})

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
ok      = sum(1 for r in results if r['ok'])
avg_t   = sum(r['time'] for r in results) / len(results)
print(f"Docs ingested:    8 PDFs ({chunk_count} chunks)")
print(f"Queries run:      {len(results)}")
print(f"Successful:       {ok}/{len(results)}")
print(f"Avg query time:   {avg_t:.1f}s")
print()
for r in results:
    status = "✅" if r['ok'] else "❌"
    print(f"  {status} Q: {r['query'][:60]}...")
    if r['sources']:
        print(f"     Sources: {', '.join(r['sources'])}")
print("=" * 80)

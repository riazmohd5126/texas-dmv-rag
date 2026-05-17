#!/usr/bin/env python3
"""
Simple Cross-Encoder RAG Server
================================
Clean, simple implementation with cross-encoder re-ranking
No normalization rules - just works!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq
from sentence_transformers import CrossEncoder
import time

# Import your RAG system
from fast_rag_universal import FastRAGSystemUniversal

app = Flask(__name__)
CORS(app)

print("="*70)
print("🚀 SIMPLE CROSS-ENCODER RAG SERVER")
print("="*70)

# =============================================================================
# CONFIGURATION - UPDATE THIS!
# =============================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCUMENTS_FOLDER = "/Users/riazmohd/Downloads/TexasRAGF/Texas_dealers_elicensing_only_scraped"

# Change this to your documents path!
# Examples:
# DOCUMENTS_FOLDER = "."  # Current folder
# DOCUMENTS_FOLDER = "/path/to/your/documents"
# =============================================================================

print("\n📂 Configuration:")
print(f"   Documents: {DOCUMENTS_FOLDER}")
print(f"   Database: ./chroma_universal")
print(f"   Port: 8080")

# Initialize Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize RAG System
print("\n🔄 Loading RAG system...")
rag = FastRAGSystemUniversal(
    groq_api_key=GROQ_API_KEY,
    collection_name="universal_rag_production",
    persist_directory="./chroma_universal",
    embedding_model="all-MiniLM-L6-v2",
    enable_profiling=False
)

# Initialize Cross-Encoder
print("🔄 Loading cross-encoder model...")
print("   (First run: downloads ~400MB model, takes 30-60s)")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
print("✅ Cross-encoder ready!")

# Check database and load documents if needed
current_count = rag.collection.count()
print(f"\n📊 Database: {current_count} chunks")

if current_count == 0:
    print("\n📁 Database empty - loading documents...")
    if os.path.exists(DOCUMENTS_FOLDER):
        print(f"   From: {DOCUMENTS_FOLDER}")
        result = rag.add_folder(DOCUMENTS_FOLDER, recursive=True)
        print(f"\n✅ Loaded!")
        print(f"   PDFs: {result.get('pdf_files', 0)}")
        print(f"   TXTs: {result.get('txt_files', 0)}")
        print(f"   Chunks: {result.get('total_chunks', 0)}")
    else:
        print(f"\n⚠️  Documents folder not found!")
        print(f"   Update DOCUMENTS_FOLDER in line 21")
else:
    print(f"✅ Using existing database")

print(f"\n🎉 System Ready! {rag.collection.count()} chunks loaded")
print("="*70 + "\n")


# =============================================================================
# MAIN QUERY FUNCTION
# =============================================================================
def query_with_cross_encoder(question: str, top_k: int = 7):
    """
    Query with cross-encoder re-ranking
    
    Pipeline:
    1. Bi-encoder: Get 50 candidates (fast, rough filter)
    2. Cross-encoder: Re-rank to best 7 (accurate, precise)
    3. LLM: Generate answer from best 7
    """
    start_time = time.time()
    
    # Step 1: Bi-encoder retrieval (get lots of candidates)
    print(f"\n📝 Query: {question}")
    print("   ⚡ Step 1: Bi-encoder retrieval...")
    
    query_embedding = rag.embedding_model.encode([question])[0]
    chroma_results = rag.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=50,  # Get 50 candidates
        include=['documents', 'metadatas', 'distances']
    )
    
    # Format candidates
    candidates = []
    for doc, meta, dist in zip(
        chroma_results['documents'][0],
        chroma_results['metadatas'][0],
        chroma_results['distances'][0]
    ):
        candidates.append({
            'content': doc,
            'metadata': meta,
            'bi_score': 1 - dist
        })
    
    retrieval_time = time.time() - start_time
    print(f"      Retrieved 50 candidates in {retrieval_time:.2f}s")
    
    if not candidates:
        return {
            'answer': "I couldn't find relevant information.",
            'sources': [],
            'query_time': time.time() - start_time
        }
    
    # Step 2: Cross-encoder re-ranking
    print("   🔄 Step 2: Cross-encoder re-ranking...")
    rerank_start = time.time()
    
    # Prepare pairs for cross-encoder
    pairs = [[question, c['content']] for c in candidates]
    
    # Get cross-encoder scores
    ce_scores = cross_encoder.predict(pairs)
    
    # Add scores and sort by cross-encoder score
    for i, candidate in enumerate(candidates):
        candidate['ce_score'] = float(ce_scores[i])
    
    # Sort by cross-encoder score (best first)
    candidates.sort(key=lambda x: x['ce_score'], reverse=True)
    top_sources = candidates[:top_k]
    
    rerank_time = time.time() - rerank_start
    print(f"      Re-ranked to top {top_k} in {rerank_time:.2f}s")
    print(f"      Best score: {top_sources[0]['ce_score']:.3f}")
    
    # Step 3: Build context
    context_parts = []
    for i, source in enumerate(top_sources, 1):
        heading = source['metadata'].get('heading', 'Unknown')
        source_file = source['metadata'].get('source_file', 'Unknown')
        file_type = source['metadata'].get('file_type', 'unknown')
        
        context_parts.append(
            f"[Source {i}] {source_file} ({file_type.upper()}) - {heading}\n"
            f"{source['content']}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    # Step 4: Generate answer with LLM
    print("   🤖 Step 3: Generating answer...")
    llm_start = time.time()
    
    prompt = f"""Answer the question using the context below.

Context:
{context}

Question: {question}

Instructions:
- Use only the context above
- Be clear and detailed
- Cite sources (e.g., "According to Source 1...")
- If information is missing, say so

Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    
    answer = response.choices[0].message.content.strip()
    llm_time = time.time() - llm_start
    total_time = time.time() - start_time
    
    print(f"      Generated in {llm_time:.2f}s")
    print(f"   ✅ Total: {total_time:.2f}s")
    
    return {
        'answer': answer,
        'sources': top_sources,
        'query_time': total_time,
        'timing': {
            'retrieval': retrieval_time,
            'reranking': rerank_time,
            'llm': llm_time
        }
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/')
def index():
    """Serve the HTML frontend"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
        <body style="font-family: Arial; padding: 40px;">
            <h1>❌ index.html not found!</h1>
            <p>Put index.html in the same folder as this script.</p>
        </body>
        </html>
        """, 404


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'documents': rag.collection.count(),
        'cross_encoder': 'enabled'
    })


@app.route('/api/query', methods=['POST'])
def query():
    """Main query endpoint"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Query with cross-encoder
        result = query_with_cross_encoder(question, top_k=7)
        
        # Format sources for frontend
        sources = []
        for source in result['sources']:
            sources.append({
                'file': source['metadata'].get('source_file', 'Unknown'),
                'file_type': source['metadata'].get('file_type', 'unknown').upper(),
                'heading': source['metadata'].get('heading', 'N/A'),
                'relevance': round(source['ce_score'] * 100, 1),
                'content': source['content'][:200] + '...'
            })
        
        return jsonify({
            'answer': result['answer'],
            'sources': sources,
            'query_time': round(result['query_time'], 2),
            'num_sources': len(sources)
        })
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    """System statistics"""
    system_stats = rag.get_stats()
    return jsonify({
        'total_documents': system_stats.get('total_documents', 0),
        'total_queries': system_stats.get('total_queries', 0),
        'avg_query_time': round(system_stats.get('avg_query_time', 0), 2),
        'pdf_files': system_stats.get('pdf_files_processed', 0),
        'txt_files': system_stats.get('txt_files_processed', 0)
    })


@app.route('/api/examples')
def examples():
    """Example questions"""
    return jsonify({
        'examples': [
            "What are the surety bond requirements?",
            "How do I apply for a dealer license?",
            "What documents are needed?",
            "What are the license fees?",
            "What is a General Distinguishing Number?"
        ]
    })


# =============================================================================
# START SERVER
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🚀 SERVER RUNNING!")
    print("="*70)
    print(f"\n   🌐 http://localhost:8080")
    print(f"   📊 {rag.collection.count()} chunks loaded")
    print(f"   🔄 Cross-encoder: ENABLED")
    print(f"\n   Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

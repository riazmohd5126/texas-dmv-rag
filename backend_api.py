#!/usr/bin/env python3
"""
Flask API Backend for RAG System
=================================

RESTful API for the Universal RAG System.
Provides endpoints for querying, document management, and stats.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from fast_rag_universal import FastRAGSystemUniversal
import logging
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCUMENTS_FOLDER = "/Users/riazmohd/Downloads/TexasRAGF 2/Texas_dealers_elicensing_only_scraped"
COLLECTION_NAME = "universal_rag_production"
DATABASE_PATH = "./chroma_universal"

# ============================================================================
# Initialize RAG System (on startup)
# ============================================================================

print("\n" + "="*80)
print("INITIALIZING RAG API SERVER")
print("="*80)

if not GROQ_API_KEY:
    print("\n❌ ERROR: GROQ_API_KEY not set!")
    print("Please run: export GROQ_API_KEY='your-key'")
    exit(1)

try:
    rag = FastRAGSystemUniversal(
        groq_api_key=GROQ_API_KEY,
        collection_name=COLLECTION_NAME,
        persist_directory=DATABASE_PATH,
        embedding_model="all-MiniLM-L6-v2",
        chunk_size=700,
        chunk_overlap=120,
        use_hybrid_search=True,
        enable_profiling=False  # Disable for API
    )
    
    # Add documents if database is empty
    if rag.collection.count() == 0:
        print("\n📁 Database empty - loading documents...")
        if os.path.exists(DOCUMENTS_FOLDER):
            result = rag.add_folder(DOCUMENTS_FOLDER, recursive=True)
            print(f"✅ Loaded {result['files_processed']} files")
        else:
            print(f"⚠️  Documents folder not found: {DOCUMENTS_FOLDER}")
    
    print("\n✅ RAG System ready!")
    print(f"   Documents: {rag.collection.count()} chunks")
    print("="*80 + "\n")

except Exception as e:
    print(f"\n❌ Failed to initialize RAG system: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('frontend', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'documents': rag.collection.count()
    })

@app.route('/api/query', methods=['POST'])
def query():
    """
    Query endpoint
    
    Request body:
    {
        "question": "Your question here",
        "top_k": 7 (optional),
        "include_sources": true (optional)
    }
    
    Response:
    {
        "answer": "...",
        "sources": [...],
        "query_time": 2.5,
        "timestamp": "..."
    }
    """
    try:
        data = request.json
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        top_k = data.get('top_k', 7)
        include_sources = data.get('include_sources', True)
        
        logger.info(f"Query: {question[:100]}...")
        
        # Query the RAG system
        response = rag.query(
            question=question,
            top_k=top_k,
            include_sources=include_sources
        )
        
        # Format sources for frontend
        sources = []
        if include_sources and 'sources' in response:
            for source in response['sources'][:7]:
                sources.append({
                    'file': source['metadata'].get('source_file', 'Unknown'),
                    'file_type': source['metadata'].get('file_type', 'unknown').upper(),
                    'heading': source['metadata'].get('heading', 'N/A'),
                    'relevance': round(source['relevance_score'] * 100, 1),
                    'content': source['content'][:200] + '...' if len(source['content']) > 200 else source['content']
                })
        
        result = {
            'answer': response['answer'],
            'sources': sources,
            'query_time': round(response['query_time'], 2),
            'num_sources': response.get('num_sources', len(sources)),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Response time: {result['query_time']}s")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    """
    Get system statistics
    
    Response:
    {
        "total_documents": 2308,
        "total_queries": 45,
        "avg_query_time": 2.8,
        ...
    }
    """
    try:
        system_stats = rag.get_stats()
        
        return jsonify({
            'total_documents': system_stats.get('total_documents', 0),
            'total_queries': system_stats.get('total_queries', 0),
            'avg_query_time': round(system_stats.get('avg_query_time', 0), 2),
            'pdf_files': system_stats.get('pdf_files_processed', 0),
            'txt_files': system_stats.get('txt_files_processed', 0),
            'collection_name': system_stats.get('collection_name', 'unknown'),
            'embedding_model': system_stats.get('embedding_model', 'unknown'),
            'hybrid_search': system_stats.get('hybrid_search', False),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def documents():
    """
    Get list of documents in the system
    
    Response:
    {
        "documents": [
            {"file": "doc1.pdf", "type": "PDF", "chunks": 142},
            ...
        ],
        "total": 12
    }
    """
    try:
        # This is a simplified version - you could extend to track actual files
        return jsonify({
            'total_chunks': rag.collection.count(),
            'message': 'Document list not implemented - shows chunk count only'
        })
    
    except Exception as e:
        logger.error(f"Documents error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def examples():
    """Get example questions"""
    return jsonify({
        'examples': [
            "What are the surety bond requirements?",
            "How do I apply for a dealer license?",
            "What documents are needed for application?",
            "What are the license fees?",
            "What is a General Distinguishing Number (GDN)?"
        ]
    })

# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n🚀 Starting Flask API Server...")
    print(f"   Frontend: http://localhost:5000")
    print(f"   API: http://localhost:5000/api/")
    print(f"   Docs: {rag.collection.count()} chunks loaded")
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=8080,
        debug=False  # Set to True for development
    )

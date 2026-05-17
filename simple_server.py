#!/usr/bin/env python3
"""
Simple RAG Web Server
=====================
Complete version with automatic document loading!
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Import your RAG system
from fast_rag_universal import FastRAGSystemUniversal

# Create Flask app
app = Flask(__name__)
CORS(app)

print("="*70)
print("INITIALIZING RAG WEB SERVER")
print("="*70)

# Initialize RAG
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ============================================================================
# IMPORTANT: UPDATE THIS PATH TO YOUR DOCUMENTS FOLDER!
# ============================================================================
DOCUMENTS_FOLDER = "/Users/riazmohd/Downloads/TexasRAGF/Texas_dealers_elicensing_only_scraped"

# Uncomment this if your PDFs/TXTs are in the same folder as this script:
# DOCUMENTS_FOLDER = "."
# ============================================================================

print("\n📂 Configuration:")
print(f"   Documents folder: {DOCUMENTS_FOLDER}")
print(f"   Database: ./chroma_universal")
print(f"   Collection: universal_rag_production")

rag = FastRAGSystemUniversal(
    groq_api_key=GROQ_API_KEY,
    collection_name="universal_rag_production",
    persist_directory="./chroma_universal",
    embedding_model="all-MiniLM-L6-v2",
    enable_profiling=False
)

# Check if database is empty and load documents
current_count = rag.collection.count()
print(f"\n📊 Current database: {current_count} chunks")

if current_count == 0:
    print("\n" + "="*70)
    print("DATABASE IS EMPTY - LOADING DOCUMENTS")
    print("="*70)
    
    if os.path.exists(DOCUMENTS_FOLDER):
        print(f"\n📁 Loading from: {DOCUMENTS_FOLDER}")
        
        # Count files first
        import glob
        pdf_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.pdf"), recursive=True)
        txt_files = glob.glob(os.path.join(DOCUMENTS_FOLDER, "**/*.txt"), recursive=True)
        total_files = len(pdf_files) + len(txt_files)
        
        print(f"   Found: {len(pdf_files)} PDF files, {len(txt_files)} TXT files")
        print(f"   Total: {total_files} files to process")
        print(f"\n⏳ This will take 5-15 minutes... please wait!\n")
        
        # Load documents
        result = rag.add_folder(DOCUMENTS_FOLDER, recursive=True)
        
        print("\n" + "="*70)
        print("✅ DOCUMENTS LOADED SUCCESSFULLY!")
        print("="*70)
        print(f"   PDF files processed: {result.get('pdf_files', 0)}")
        print(f"   TXT files processed: {result.get('txt_files', 0)}")
        print(f"   Total chunks created: {result.get('total_chunks', 0)}")
    else:
        print("\n" + "="*70)
        print("⚠️  ERROR: DOCUMENTS FOLDER NOT FOUND")
        print("="*70)
        print(f"   Looking for: {DOCUMENTS_FOLDER}")
        print(f"\n   Please update DOCUMENTS_FOLDER in simple_server.py (line 27)")
        print(f"   Set it to the path where your PDF/TXT files are located.")
        print("="*70)
else:
    print(f"✅ Using existing database with {current_count} chunks")

final_count = rag.collection.count()
print(f"\n🎉 RAG System Ready! Total chunks: {final_count}")

if final_count == 0:
    print("\n⚠️  WARNING: No documents loaded!")
    print("   Update DOCUMENTS_FOLDER path and restart server.")

print("="*70 + "\n")

# Serve the HTML
@app.route('/')
def index():
    """Serve the frontend HTML"""
    try:
        with open('index.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        try:
            with open('frontend/index.html', 'r') as f:
                return f.read()
        except FileNotFoundError:
            return f"""
            <html>
            <head><title>Error</title></head>
            <body style="font-family: Arial; padding: 40px;">
                <h1>❌ Error: index.html not found!</h1>
                <p><b>Looking for index.html in:</b></p>
                <ul>
                    <li>Same folder as simple_server.py</li>
                    <li>OR inside frontend/ subfolder</li>
                </ul>
                <p><b>Current directory:</b> {os.getcwd()}</p>
                <p><b>Files here:</b></p>
                <pre>{chr(10).join(os.listdir('.'))}</pre>
                <hr>
                <p><b>Solution:</b> Download index.html and put it in the same folder as simple_server.py</p>
            </body>
            </html>
            """, 404

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'documents': rag.collection.count()
    })

@app.route('/api/query', methods=['POST'])
def query():
    """Query endpoint"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"\n📝 Query received: {question[:100]}...")
        
        # Query RAG
        response = rag.query(
            question=question,
            top_k=7,
            include_sources=True
        )
        
        print(f"✅ Query completed in {response.get('query_time', 0):.2f}s")
        
        # Format sources
        sources = []
        for source in response.get('sources', [])[:7]:
            sources.append({
                'file': source['metadata'].get('source_file', 'Unknown'),
                'file_type': source['metadata'].get('file_type', 'unknown').upper(),
                'heading': source['metadata'].get('heading', 'N/A'),
                'relevance': round(source['relevance_score'] * 100, 1),
                'content': source['content'][:200] + '...'
            })
        
        return jsonify({
            'answer': response['answer'],
            'sources': sources,
            'query_time': round(response['query_time'], 2),
            'num_sources': len(sources)
        })
    
    except Exception as e:
        print(f"\n❌ Query Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/stats')
def stats():
    """System stats"""
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
            "What documents are needed for application?",
            "What are the license fees?",
            "What is a General Distinguishing Number (GDN)?"
        ]
    })

if __name__ == '__main__':
    print("="*70)
    print("🚀 RAG WEB SERVER RUNNING!")
    print("="*70)
    print(f"\n   🌐 Open in browser: http://localhost:8080")
    print(f"   📊 Documents loaded: {rag.collection.count()}")
    print(f"\n   Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

#!/usr/bin/env python3
"""
Enhanced Simple Server with Query Rewriting
============================================
Automatically simplifies complex queries for better retrieval
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from groq import Groq

# Import your RAG system
from fast_rag_universal import FastRAGSystemUniversal

app = Flask(__name__)
CORS(app)

print("="*70)
print("INITIALIZING ENHANCED RAG WEB SERVER")
print("="*70)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DOCUMENTS_FOLDER = "/Users/riazmohd/Downloads/TexasRAGF/Texas_dealers_elicensing_only_scraped"

# Initialize Groq for query rewriting
groq_client = Groq(api_key=GROQ_API_KEY)

print("\n📂 Configuration:")
print(f"   Documents folder: {DOCUMENTS_FOLDER}")
print(f"   Database: ./chroma_universal")
print(f"   🆕 Query Rewriting: ENABLED")

rag = FastRAGSystemUniversal(
    groq_api_key=GROQ_API_KEY,
    collection_name="universal_rag_production",
    persist_directory="./chroma_universal",
    embedding_model="all-MiniLM-L6-v2",
    enable_profiling=False
)

current_count = rag.collection.count()
print(f"\n📊 Current database: {current_count} chunks")

if current_count == 0:
    print("\n📁 Database is empty - loading documents...")
    if os.path.exists(DOCUMENTS_FOLDER):
        print(f"   From: {DOCUMENTS_FOLDER}")
        result = rag.add_folder(DOCUMENTS_FOLDER, recursive=True)
        print(f"\n✅ Successfully loaded documents!")
        print(f"   Total chunks: {result.get('total_chunks', 0)}")
    else:
        print(f"\n⚠️  Documents folder not found: {DOCUMENTS_FOLDER}")
else:
    print(f"✅ Using existing database with {current_count} chunks")

print(f"\n🎉 RAG System Ready! Total chunks: {rag.collection.count()}")
print("="*70 + "\n")


def normalize_query(query: str) -> str:
    """
    Normalize query to handle word variations
    """
    normalized = query
    
    # Common word variations that should be normalized
    normalizations = {
        # license variations
        ' license ': ' licensing ',
        ' license?': ' licensing?',
        'license requirements': 'licensing requirements',
        
        # application variations
        ' application ': ' apply ',
        ' applications ': ' apply ',
        
        # requirement variations  
        ' requirement ': ' requirements ',
        
        # Common abbreviations
        'GDN': 'General Distinguishing Number',
        'MVD': 'Motor Vehicle Department',
    }
    
    original = normalized
    for old, new in normalizations.items():
        if old in normalized:
            normalized = normalized.replace(old, new)
    
    if normalized != original:
        print(f"   📝 Normalized:")
        print(f"      '{original}' → '{normalized}'")
    
    return normalized


def rewrite_query(original_query: str) -> str:
    """
    Rewrite complex queries into simpler, more retrievable forms
    """
    # Check if query is complex (has words like "explain", "describe", "analyze")
    complex_keywords = ['explain', 'describe', 'analyze', 'detail', 'elaborate', 
                       'discuss', 'key features', 'main points']
    
    # Also check for redundant legal/descriptive terms that confuse retrieval
    redundant_terms = ['law', 'rule', 'regulation', 'statute', 'policy', 'code']
    
    is_complex = any(keyword in original_query.lower() for keyword in complex_keywords)
    
    # Also trigger rewrite if query has redundant terms at the end
    # e.g., "What is House Bill 718 law?" -> remove "law"
    has_redundant = any(original_query.lower().strip().endswith(term) or 
                       original_query.lower().strip().endswith(term + '?')
                       for term in redundant_terms)
    
    if not is_complex and not has_redundant:
        return original_query  # Already simple and clean
    
    print(f"   🔄 Rewriting query...")
    if is_complex:
        print(f"      Reason: Contains complex keywords")
    if has_redundant:
        print(f"      Reason: Contains redundant terms")
    
    # Use LLM to rewrite
    rewrite_prompt = f"""
Rewrite this query into a simple, direct question that would better retrieve relevant information:

Original: "{original_query}"

Rules:
1. Keep it short and direct (under 10 words)
2. Focus on the main subject/entity
3. Use "What is" or "What are" format
4. Remove words like "explain", "describe", "key features"
5. Remove redundant legal terms like "law", "rule", "regulation" at the end
6. Keep the core entity/subject (like "House Bill 718")

Examples:
- "What is House Bill 718 law?" → "What is House Bill 718?"
- "Explain the dealer licensing regulation" → "What is dealer licensing?"
- "Describe the surety bond rule" → "What is a surety bond?"

Rewritten query (just the question, nothing else):"""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.1,
            max_tokens=50
        )
        
        rewritten = response.choices[0].message.content.strip()
        # Remove quotes if present
        rewritten = rewritten.strip('"').strip("'")
        
        print(f"   Original: {original_query}")
        print(f"   Rewritten: {rewritten}")
        
        return rewritten
    
    except Exception as e:
        print(f"   ⚠️  Rewrite failed, using original: {e}")
        return original_query


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
            <body style="font-family: Arial; padding: 40px;">
                <h1>❌ Error: index.html not found!</h1>
                <p>Current directory: {os.getcwd()}</p>
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
    """Query endpoint with query rewriting"""
    try:
        data = request.json
        original_question = data.get('question', '')
        
        if not original_question:
            return jsonify({'error': 'No question provided'}), 400
        
        print(f"\n📝 Query received: {original_question}")
        
        # Step 1: Normalize query (handle word variations)
        normalized_question = normalize_query(original_question)
        
        # Step 2: Rewrite complex queries
        search_question = rewrite_query(normalized_question)
        
        # Query RAG (using rewritten query for retrieval)
        response = rag.query(
            question=search_question,  # Use rewritten for retrieval
            top_k=7,
            include_sources=True
        )
        
        # But use original question for answer generation
        if search_question != original_question:
            # Re-generate answer with original question
            context_parts = []
            for i, source in enumerate(response.get('sources', [])[:7], 1):
                heading = source['metadata'].get('heading', 'Unknown section')
                source_file = source['metadata'].get('source_file', 'Unknown')
                file_type = source['metadata'].get('file_type', 'unknown')
                
                context_parts.append(
                    f"[Source {i}] From '{source_file}' ({file_type.upper()}), Section: {heading}\n"
                    f"{source['content']}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Generate answer for ORIGINAL question
            prompt = f"""
You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question:
{original_question}

Instructions:
1. Answer the question using the context above
2. Be comprehensive and detailed
3. Cite sources when possible
4. If context is insufficient, state what's missing

Answer:"""
            
            llm_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000  # Increased for detailed answers
            )
            
            response['answer'] = llm_response.choices[0].message.content.strip()
        
        print(f"✅ Query completed in {response.get('query_time', 0):.2f}s")
        
        # Format sources
        sources = []
        for source in response.get('sources', [])[:7]:
            sources.append({
                'file': source['metadata'].get('source_file', 'Unknown'),
                'file_type': source['metadata'].get('file_type', 'unknown').upper(),
                'heading': source['metadata'].get('heading', 'N/A'),
                'relevance': round(source.get('relevance_score', source.get('score', 0)) * 100, 1),
                'content': source['content'][:200] + '...'
            })
        
        return jsonify({
            'answer': response['answer'],
            'sources': sources,
            'query_time': round(response['query_time'], 2),
            'num_sources': len(sources),
            'original_query': original_question,
            'search_query': search_question if search_question != original_question else None
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
    print("🚀 ENHANCED RAG WEB SERVER RUNNING!")
    print("="*70)
    print(f"\n   🌐 Open in browser: http://localhost:8080")
    print(f"   📊 Documents loaded: {rag.collection.count()}")
    print(f"   🆕 Query Rewriting: ENABLED")
    print(f"\n   Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=8080, debug=False)

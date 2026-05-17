#!/usr/bin/env python3
"""
Run Complete Advanced RAG System
=================================

Simple script to process documents and query your RAG system.

Usage:
    python3 run_complete_rag.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from complete_rag_app import AdvancedRAGSystem


def main():
    """Main function - interactive setup and usage"""
    
    print("\n" + "="*80)
    print("COMPLETE ADVANCED RAG SYSTEM")
    print("="*80)
    print("\nFeatures:")
    print("  ✅ Structure-aware chunking (splits by headings)")
    print("  ✅ Hybrid retrieval (semantic + keyword search)")
    print("  ✅ Re-ranking (improves result quality)")
    print("  ✅ Rich metadata (source attribution)")
    print("="*80)
    
    # Step 1: Check API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("\n❌ Error: GROQ_API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export GROQ_API_KEY='your-key-here'")
        return
    
    print("\n✅ API key found")
    
    # Step 2: Initialize system
    print("\n" + "="*80)
    print("INITIALIZATION")
    print("="*80)
    
    print("\nInitializing RAG system with optimal settings...")
    
    rag = AdvancedRAGSystem(
        groq_api_key=GROQ_API_KEY,
        collection_name="advanced_rag_system",
        persist_directory="./chroma_advanced_rag",
        embedding_model="all-mpnet-base-v2",  # High quality embeddings
        llm_model="llama-3.3-70b-versatile",   # Fast and capable
        chunk_size=700,                         # Good for regulatory docs
        chunk_overlap=150,                      # High overlap for context
        use_hybrid_search=True,                 # Enable hybrid retrieval
        use_reranking=True                      # Enable re-ranking
    )
    
    # Step 3: Check if we need to add documents
    if rag.collection.count() == 0:
        print("\n" + "="*80)
        print("DOCUMENT INGESTION")
        print("="*80)
        
        print("\n📁 Your database is empty. Let's add documents!")
        print("\nEnter the path to your documents folder:")
        print("  Example: /Users/riazmohd/Documents/TexasDMV")
        
        folder_path = input("\nFolder path: ").strip()
        
        if not folder_path or not Path(folder_path).exists():
            print(f"\n❌ Folder '{folder_path}' not found!")
            print("\nYou can add documents later by running:")
            print("  python3 -c 'from complete_rag_app import AdvancedRAGSystem; ...")
            return
        
        print(f"\n📥 Processing documents from: {folder_path}")
        print("This may take a few minutes...")
        
        result = rag.add_folder(
            folder_path=folder_path,
            recursive=True,
            additional_metadata={
                'category': 'regulatory',
                'source_type': 'user_documents'
            }
        )
        
        if result['files_processed'] == 0:
            print("\n⚠️  No documents were processed!")
            return
    else:
        print(f"\n✅ Database contains {rag.collection.count()} chunks")
    
    # Step 4: Interactive query mode
    print("\n" + "="*80)
    print("INTERACTIVE QUERY MODE")
    print("="*80)
    
    print("\nYou can now ask questions about your documents!")
    print("Commands:")
    print("  • Type your question and press Enter")
    print("  • Type 'stats' to see statistics")
    print("  • Type 'quit' or 'exit' to exit")
    print("="*80)
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\n❓ Your question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() == 'stats':
            print("\n📊 System Statistics:")
            print("-"*80)
            stats = rag.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            continue
        
        # Query the system
        try:
            response = rag.query(
                question=user_input,
                top_k=10,
                include_sources=True
            )
            
            # Display answer
            print("\n" + "="*80)
            print("ANSWER")
            print("="*80)
            print(f"\n{response['answer']}")
            
            # Display sources
            print("\n" + "="*80)
            print(f"SOURCES (Top 3 of {response['num_sources']})")
            print("="*80)
            
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"\n📄 Source {i} (Relevance: {source['relevance_score']:.1%})")
                print(f"   File: {source['metadata'].get('source_file', 'Unknown')}")
                
                heading = source['metadata'].get('heading')
                if heading:
                    print(f"   Section: {heading}")
                
                section_num = source['metadata'].get('section_number')
                if section_num:
                    print(f"   Section number: {section_num}")
                
                print(f"   Method: {source['retrieval_method']}")
                print(f"   Preview: {source['content'][:200]}...")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final statistics
    print("\n" + "="*80)
    print("SESSION SUMMARY")
    print("="*80)
    
    stats = rag.get_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Queries in session: {stats['total_queries']}")
    print(f"   Collection: {stats['collection_name']}")
    
    print("\n" + "="*80)
    print("✅ SESSION COMPLETE")
    print("="*80)
    
    print("\n💡 To use this system in your code:")
    print("""
from complete_rag_app import AdvancedRAGSystem

rag = AdvancedRAGSystem(
    groq_api_key="your-key",
    collection_name="my_docs",
    use_hybrid_search=True,
    use_reranking=True
)

# Add documents
rag.add_folder("./documents")

# Query
response = rag.query("Your question?")
print(response['answer'])
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Universal RAG System - PDF + TXT Support
=========================================

Supports BOTH file types:
- PDF files (.pdf)
- Text files (.txt)

Same fast performance with universal file support!
"""

import os
from fast_rag_universal import FastRAGSystemUniversal

def main():
    # ========================================================================
    # CONFIGURATION - CHANGE THESE VALUES
    # ========================================================================
    
    # Your documents folder path (can contain PDF and TXT files)
    DOCUMENTS_FOLDER = "/Users/riazmohd/Downloads/TexasRAGF/Texas_dealers_elicensing_only_scraped"  # ← CHANGE THIS!
    
    # Database name and location
    COLLECTION_NAME = "universal_rag_production"
    DATABASE_PATH = "./chroma_universal"
    
    # ========================================================================
    # END CONFIGURATION
    # ========================================================================
    
    print("="*80)
    print("UNIVERSAL RAG SYSTEM (PDF + TXT Support)")
    print("="*80)
    
    # Check API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("\n❌ Error: GROQ_API_KEY not set!")
        print("\nPlease run:")
        print("  export GROQ_API_KEY='your-groq-api-key-here'")
        return
    
    print("\n✅ API key found")
    
    # Initialize Universal System
    print("\n" + "="*80)
    print("INITIALIZATION")
    print("="*80)
    print("\n⚡ Initializing UNIVERSAL system...")
    
    rag = FastRAGSystemUniversal(
        groq_api_key=GROQ_API_KEY,
        collection_name=COLLECTION_NAME,
        persist_directory=DATABASE_PATH,
        embedding_model="all-MiniLM-L6-v2",     # Fast
        chunk_size=700,
        chunk_overlap=120,
        use_hybrid_search=True,
        enable_profiling=True
    )
    
    print("\n✅ System initialized!")
    print(f"\n📄 Supported file types:")
    print(f"   • PDF files (.pdf)")
    print(f"   • Text files (.txt)")
    print(f"   • Both will be chunked the same way!")
    
    # Check database
    current_docs = rag.collection.count()
    print(f"\n📊 Current database:")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Chunks: {current_docs}")
    
    # Add documents if needed
    if current_docs == 0:
        print("\n" + "="*80)
        print("DOCUMENT INGESTION")
        print("="*80)
        
        if not os.path.exists(DOCUMENTS_FOLDER):
            print(f"\n❌ Error: Folder not found!")
            print(f"   Looking for: {DOCUMENTS_FOLDER}")
            print(f"\n💡 Please update DOCUMENTS_FOLDER in the script")
            return
        
        print(f"\n📁 Processing documents from:")
        print(f"   {DOCUMENTS_FOLDER}")
        print(f"\n⏳ Looking for PDF and TXT files...")
        
        result = rag.add_folder(
            folder_path=DOCUMENTS_FOLDER,
            recursive=True,
            additional_metadata={'category': 'production'}
        )
        
        if result['files_processed'] == 0:
            print("\n❌ No documents were processed!")
            return
        
        print(f"\n✅ Successfully processed documents!")
        print(f"   PDF files: {result['pdf_files']}")
        print(f"   TXT files: {result['txt_files']}")
        print(f"   Total files: {result['files_processed']}")
        print(f"   Total chunks: {result['total_chunks']}")
    
    # Interactive Query Mode
    print("\n" + "="*80)
    print("INTERACTIVE QUERY MODE")
    print("="*80)
    print("\nYou can now ask questions about your documents!")
    print("(Questions will search across BOTH PDF and TXT files)")
    print("\nCommands:")
    print("  • Type your question")
    print("  • Type 'stats' for statistics")
    print("  • Type 'quit' to exit")
    print("="*80)
    
    query_count = 0
    
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
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
            continue
        
        # Process query
        query_count += 1
        
        try:
            response = rag.query(
                question=user_input,
                top_k=7,
                include_sources=True
            )
            
            # Display answer
            print("\n" + "="*80)
            print(f"ANSWER (Query #{query_count})")
            print("="*80)
            print(f"\n{response['answer']}")
            
            # Display performance
            print("\n" + "="*80)
            print("PERFORMANCE")
            print("="*80)
            print(f"\n⏱️  Total time: {response['query_time']:.2f}s")
            
            if response.get('timing_breakdown'):
                print("\nBreakdown:")
                for component, duration in response['timing_breakdown'].items():
                    pct = (duration / response['query_time']) * 100
                    print(f"  • {component}: {duration:.3f}s ({pct:.1f}%)")
            
            # Display sources
            print("\n" + "="*80)
            print(f"SOURCES (Top 3 of {response['num_sources']})")
            print("="*80)
            
            for i, source in enumerate(response['sources'][:3], 1):
                file_type = source['metadata'].get('file_type', 'unknown').upper()
                
                print(f"\n📄 Source {i} ({file_type} file)")
                print(f"   Relevance: {source['relevance_score']:.1%}")
                print(f"   File: {source['metadata'].get('source_file', 'Unknown')}")
                
                heading = source['metadata'].get('heading')
                if heading:
                    print(f"   Section: {heading}")
                
                print(f"   Preview: {source['content'][:120]}...")
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("SESSION SUMMARY")
    print("="*80)
    
    stats = rag.get_stats()
    print(f"\n📊 Statistics:")
    print(f"   Queries: {query_count}")
    print(f"   Average time: {stats['avg_query_time']:.2f}s")
    print(f"   PDF files: {stats['pdf_files_processed']}")
    print(f"   TXT files: {stats['txt_files_processed']}")
    print(f"   Total chunks: {stats['total_documents']}")
    
    print("\n✅ Session complete!")
    
    print("\n💡 Both PDF and TXT files were searchable!")
    print("   The system treated them exactly the same way.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

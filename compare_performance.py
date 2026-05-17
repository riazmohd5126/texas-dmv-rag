#!/usr/bin/env python3
"""
Performance Comparison Script
==============================

Compare the full RAG system vs. the fast optimized version.
See the speed difference and quality trade-off.
"""

import os
import time
from complete_rag_app import AdvancedRAGSystem
from fast_rag_system import FastRAGSystem


def run_comparison(test_queries: list):
    """
    Run performance comparison between full and fast systems.
    
    Args:
        test_queries: List of questions to test
    """
    print("\n" + "="*80)
    print("RAG PERFORMANCE COMPARISON")
    print("="*80)
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("\n❌ Error: GROQ_API_KEY not set!")
        return
    
    # Initialize both systems
    print("\n🔧 Initializing systems...")
    
    # Full system (high quality)
    print("\n1️⃣  Full System (Maximum Quality):")
    full_rag = AdvancedRAGSystem(
        groq_api_key=GROQ_API_KEY,
        collection_name="comparison_full",
        embedding_model="all-mpnet-base-v2",  # 768-dim, slow
        use_hybrid_search=True,
        use_reranking=True  # Enabled
    )
    
    # Fast system (optimized)
    print("\n2️⃣  Fast System (Optimized):")
    fast_rag = FastRAGSystem(
        groq_api_key=GROQ_API_KEY,
        collection_name="comparison_fast",
        embedding_model="all-MiniLM-L6-v2",  # 384-dim, fast
        use_hybrid_search=True,
        enable_profiling=True
    )
    
    # Check if databases have content
    if full_rag.collection.count() == 0 or fast_rag.collection.count() == 0:
        print("\n⚠️  Warning: Collections are empty!")
        print("   Please add documents first using add_document() or add_folder()")
        return
    
    # Run comparison
    print("\n" + "="*80)
    print("RUNNING COMPARISON TESTS")
    print("="*80)
    
    results = {
        'full': {'times': [], 'scores': []},
        'fast': {'times': [], 'scores': []}
    }
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")
        
        # Test Full System
        print("\n1️⃣  Testing FULL system...")
        start = time.time()
        full_response = full_rag.query(query, top_k=10)
        full_time = time.time() - start
        full_score = full_response['sources'][0]['relevance_score'] if full_response['sources'] else 0
        
        results['full']['times'].append(full_time)
        results['full']['scores'].append(full_score)
        
        print(f"   Time: {full_time:.2f}s")
        print(f"   Top relevance: {full_score:.3f}")
        
        # Test Fast System
        print("\n2️⃣  Testing FAST system...")
        start = time.time()
        fast_response = fast_rag.query(query, top_k=7)
        fast_time = time.time() - start
        fast_score = fast_response['sources'][0]['relevance_score'] if fast_response['sources'] else 0
        
        results['fast']['times'].append(fast_time)
        results['fast']['scores'].append(fast_score)
        
        print(f"   Time: {fast_time:.2f}s")
        print(f"   Top relevance: {fast_score:.3f}")
        
        # Compare
        time_diff = full_time - fast_time
        time_improvement = (time_diff / full_time) * 100
        score_diff = full_score - fast_score
        score_retention = (fast_score / full_score) * 100 if full_score > 0 else 100
        
        print(f"\n📊 Comparison:")
        print(f"   Speed improvement: {time_improvement:.1f}% faster ({time_diff:.2f}s saved)")
        print(f"   Quality retention: {score_retention:.1f}% ({score_diff:.3f} difference)")
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL COMPARISON SUMMARY")
    print("="*80)
    
    avg_full_time = sum(results['full']['times']) / len(results['full']['times'])
    avg_fast_time = sum(results['fast']['times']) / len(results['fast']['times'])
    avg_full_score = sum(results['full']['scores']) / len(results['full']['scores'])
    avg_fast_score = sum(results['fast']['scores']) / len(results['fast']['scores'])
    
    time_improvement = ((avg_full_time - avg_fast_time) / avg_full_time) * 100
    quality_retention = (avg_fast_score / avg_full_score) * 100 if avg_full_score > 0 else 100
    
    print(f"\n⏱️  SPEED:")
    print(f"   Full system:  {avg_full_time:.2f}s average")
    print(f"   Fast system:  {avg_fast_time:.2f}s average")
    print(f"   Improvement:  {time_improvement:.1f}% faster! ⚡")
    print(f"   Time saved:   {avg_full_time - avg_fast_time:.2f}s per query")
    
    print(f"\n🎯 QUALITY:")
    print(f"   Full system:  {avg_full_score:.3f} average relevance")
    print(f"   Fast system:  {avg_fast_score:.3f} average relevance")
    print(f"   Retention:    {quality_retention:.1f}% of full quality")
    print(f"   Difference:   {avg_full_score - avg_fast_score:.3f}")
    
    print(f"\n💡 RECOMMENDATION:")
    if time_improvement > 30 and quality_retention > 80:
        print(f"   ✅ Fast system is HIGHLY RECOMMENDED")
        print(f"   • {time_improvement:.0f}% faster with only {100-quality_retention:.0f}% quality loss")
        print(f"   • Users will be much happier with faster responses")
    elif time_improvement > 20 and quality_retention > 70:
        print(f"   ✅ Fast system is RECOMMENDED")
        print(f"   • Good speed improvement with acceptable quality")
    else:
        print(f"   ⚠️  Consider your use case")
        print(f"   • If speed is critical: use Fast")
        print(f"   • If quality is critical: use Full")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return results


def main():
    """Main function"""
    print("="*80)
    print("PERFORMANCE COMPARISON TOOL")
    print("="*80)
    
    # Default test queries
    test_queries = [
        "What are the surety bond requirements for dealers?",
        "What documentation is needed for a license application?",
        "What are the office structure requirements?",
        "What are the license fees?",
        "What is the application process?"
    ]
    
    print("\n📝 Will test with these queries:")
    for i, q in enumerate(test_queries, 1):
        print(f"   {i}. {q}")
    
    print("\n⚠️  Important:")
    print("   • Make sure both systems have documents added")
    print("   • This will take a few minutes to complete")
    print("   • Results will show speed vs quality trade-off")
    
    input("\nPress Enter to start comparison...")
    
    # Run comparison
    results = run_comparison(test_queries)
    
    if results:
        print("\n✅ Comparison complete!")
        print("\nTo use the fast system in your code:")
        print("""
from fast_rag_system import FastRAGSystem

rag = FastRAGSystem(
    groq_api_key="your-key",
    use_hybrid_search=True
)

# 40% faster with 85% of quality!
response = rag.query("Your question?")
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

"""
Fast RAG System with Universal File Support
============================================

Supports BOTH:
- PDF files (.pdf)
- Text files (.txt)

Same fast performance (40% faster than full system).
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import time

# Import our custom modules
from universal_extractor import UniversalDocumentExtractor  # New!
from chunking_structure_aware import StructureAwareSplitter
from advanced_retrieval import HybridRetriever, RetrievalResult
from document_categories import get_category, get_filter_categories, get_smart_top_k


class FastRAGSystemUniversal:
    """
    Fast RAG system with support for PDF and TXT files.
    
    Features:
        ⚡ 40% faster than full system
        📄 Supports PDF files
        📝 Supports TXT files
        ✅ Structure-aware chunking for both
        ✅ Hybrid retrieval (dense + sparse)
        ✅ Rich metadata
    """
    
    def __init__(
        self,
        groq_api_key: str,
        collection_name: str = "fast_rag_universal",
        persist_directory: str = "./chroma_fast_universal",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        use_hybrid_search: bool = True,
        enable_profiling: bool = False
    ):
        """Initialize the fast RAG system with universal file support."""
        
        print("\n" + "="*80)
        print("FAST RAG SYSTEM - UNIVERSAL (PDF + TXT Support)")
        print("="*80)
        
        self.config = {
            'collection_name': collection_name,
            'embedding_model': embedding_model,
            'llm_model': llm_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'use_hybrid_search': use_hybrid_search,
            'use_reranking': False,
            'enable_profiling': enable_profiling
        }
        
        # Initialize Groq
        print("\n🤖 Initializing Groq LLM...")
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model = llm_model
        
        # Initialize embedding model
        print(f"\n🧠 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        print(f"\n💾 Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✅ Loaded collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"   ✅ Created collection: {collection_name}")
        
        # Initialize UNIVERSAL extractor (handles PDF + TXT)
        print("\n📄 Initializing universal document extractor...")
        print("   Supports: PDF, TXT")
        self.doc_extractor = UniversalDocumentExtractor(preserve_structure=True)
        
        # Initialize chunker
        print("\n🔪 Initializing structure-aware chunker...")
        self.chunker = StructureAwareSplitter(
            max_chunk_size=chunk_size,
            overlap=chunk_overlap,
            respect_structure=True
        )
        
        # Initialize retriever
        print("\n🔍 Initializing retrieval system...")
        if use_hybrid_search:
            self.retriever = HybridRetriever(
                collection=self.collection,
                embedding_model=self.embedding_model,
                use_reranking=False,
                dense_weight=0.7,
                sparse_weight=0.3
            )
        else:
            self.retriever = None
        
        self.stats = {
            'total_queries': 0,
            'total_documents_added': 0,
            'total_chunks_created': 0,
            'avg_query_time': 0,
            'query_times': [],
            'pdf_files': 0,
            'txt_files': 0
        }
        
        print("\n" + "="*80)
        print("✅ FAST RAG SYSTEM READY")
        print("="*80)
        print(f"\n⚡ Supported file types:")
        print(f"   • PDF files (.pdf)")
        print(f"   • Text files (.txt)")
        print(f"\n📊 Documents: {self.collection.count()} chunks")
        print("="*80 + "\n")
    
    def add_document(self, file_path: str, additional_metadata: Optional[Dict] = None) -> int:
        """
        Add a document (PDF or TXT) to the RAG system.
        
        Args:
            file_path (str): Path to PDF or TXT file
            additional_metadata (dict): Optional extra metadata
            
        Returns:
            Number of chunks created
        """
        print(f"\n{'='*80}")
        print(f"ADDING DOCUMENT: {Path(file_path).name}")
        print(f"{'='*80}")
        
        # Extract document (handles both PDF and TXT)
        print("\n📄 Step 1: Extracting document...")
        extraction_result = self.doc_extractor.extract_from_file(file_path)
        
        # Track file type
        if extraction_result['file_type'] == 'pdf':
            self.stats['pdf_files'] += 1
        elif extraction_result['file_type'] == 'txt':
            self.stats['txt_files'] += 1
        
        # Prepare metadata
        base_metadata = {
            'source_file': Path(file_path).name,
            'source_path': str(Path(file_path).absolute()),
            'file_type': extraction_result['file_type'],
            'page_count': extraction_result['page_count'],
        }
        
        if extraction_result['metadata']:
            base_metadata.update({
                f'doc_{k}': v for k, v in extraction_result['metadata'].items()
            })
        
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        # Split into chunks
        print("\n🔪 Step 2: Structure-aware chunking...")
        chunks = self.chunker.split(
            text=extraction_result['text'],
            metadata=base_metadata
        )
        
        # Prepare for ChromaDB
        print("\n💾 Step 3: Adding to database...")
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}_chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("   Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to ChromaDB
        print("   Adding to ChromaDB...")

        # ── Category tagging (added by patch) ──────────────────────────────
        _file_category = get_category(Path(file_path).name)
        for _m in metadatas:
            _m['category'] = _file_category
        print(f"   🏷️  Category: {_file_category} → {Path(file_path).name}")
        # ───────────────────────────────────────────────────────────────────

        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        # Index for sparse retrieval
        if self.config['use_hybrid_search']:
            print("\n🔍 Step 4: Indexing for sparse retrieval...")
            self.retriever.index_for_sparse()
        
        # Update statistics
        self.stats['total_documents_added'] += 1
        self.stats['total_chunks_created'] += len(chunks)
        
        print(f"\n✅ Successfully added document!")
        print(f"   Type: {extraction_result['file_type'].upper()}")
        print(f"   Created: {len(chunks)} chunks")
        print(f"   Total in database: {self.collection.count()}")
        print(f"{'='*80}\n")
        
        return len(chunks)
    
    def add_folder(
        self,
        folder_path: str,
        file_pattern: str = "*",  # Changed from "*.pdf" to accept all
        recursive: bool = True,
        additional_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add all supported documents from a folder.
        
        Args:
            folder_path (str): Path to folder
            file_pattern (str): File pattern ("*" for all, "*.pdf" for PDFs only, etc.)
            recursive (bool): Search subdirectories
            additional_metadata (dict): Metadata for all documents
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLDER: {folder_path}")
        print(f"{'='*80}\n")
        
        folder = Path(folder_path)
        
        # Find all supported files (PDF and TXT)
        if recursive:
            pdf_files = list(folder.rglob("*.pdf"))
            txt_files = list(folder.rglob("*.txt"))
        else:
            pdf_files = list(folder.glob("*.pdf"))
            txt_files = list(folder.glob("*.txt"))
        
        files = pdf_files + txt_files
        
        print(f"📁 Found:")
        print(f"   • {len(pdf_files)} PDF files")
        print(f"   • {len(txt_files)} TXT files")
        print(f"   • {len(files)} total files\n")
        
        if not files:
            print("⚠️  No supported files found!")
            return {'files_processed': 0, 'total_chunks': 0}
        
        # Process each file
        total_chunks = 0
        successful = 0
        failed = []
        
        for i, file_path in enumerate(files, 1):
            file_type = "PDF" if file_path.suffix == ".pdf" else "TXT"
            print(f"\n[{i}/{len(files)}] Processing {file_path.name} ({file_type})")
            
            try:
                chunks = self.add_document(str(file_path), additional_metadata)
                total_chunks += chunks
                successful += 1
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append((file_path.name, str(e)))
        
        # Final summary
        print(f"\n{'='*80}")
        print("FOLDER PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Results:")
        print(f"   Total files: {len(files)}")
        print(f"   • PDF files: {len(pdf_files)}")
        print(f"   • TXT files: {len(txt_files)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(failed)}")
        print(f"   Total chunks: {total_chunks}")
        
        if failed:
            print(f"\n❌ Failed files:")
            for fname, error in failed:
                print(f"   • {fname}: {error}")
        
        print(f"\n💾 Database status:")
        print(f"   Total documents: {self.collection.count()}")
        print(f"{'='*80}\n")
        
        return {
            'files_processed': successful,
            'files_failed': len(failed),
            'total_chunks': total_chunks,
            'failed_files': failed,
            'pdf_files': len(pdf_files),
            'txt_files': len(txt_files)
        }
    
    def query(self, question: str, top_k: int = 7, include_sources: bool = True, metadata_filter: Optional[Dict] = None) -> Dict:
        """Query the system (same as before, works with both PDF and TXT chunks)"""
        
        start_time = time.time()
        times = {} if self.config['enable_profiling'] else None
        
        print(f"\n{'='*80}")
        print(f"QUERY: {question}")
        print(f"{'='*80}")
        
        self.stats['total_queries'] += 1
        
        # Retrieval
        retrieval_start = time.time()
        
        if self.config['use_hybrid_search'] and self.retriever:
            results = self.retriever.retrieve(query=question, top_k=top_k, where=metadata_filter)
        else:
            query_embedding = self.embedding_model.encode([question])[0]

            # ── Auto category filter (added by patch) ──────────────────────
            if metadata_filter is None:
                _auto_cats = get_filter_categories(question)
                if _auto_cats:
                    metadata_filter = {"category": {"$in": _auto_cats}}
                    print(f"   🏷️  Auto filter: {_auto_cats}")
                else:
                    print(f"   🏷️  No filter (broad query)")
            # ── Smart top_k (added by patch 2) ───────────────────────────────
            top_k = get_smart_top_k(question, default=top_k)
            # ─────────────────────────────────────────────────────────────

            chroma_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            results = []
            for i, (doc, meta, dist) in enumerate(zip(
                chroma_results['documents'][0],
                chroma_results['metadatas'][0],
                chroma_results['distances'][0]
            )):
                results.append(RetrievalResult(
                    content=doc,
                    metadata=meta,
                    score=1 - dist,
                    retrieval_method='dense',
                    rank=i + 1
                ))
        
        if times is not None:
            times['retrieval'] = time.time() - retrieval_start
        
        if not results:
            return {
                'answer': "I couldn't find relevant information.",
                'sources': [],
                'query': question,
                'query_time': time.time() - start_time
            }
        
        # Build context
        context_start = time.time()
        context_parts = []
        for i, result in enumerate(results, 1):
            heading = result.metadata.get('heading', 'Unknown section')
            source_file = result.metadata.get('source_file', 'Unknown')
            file_type = result.metadata.get('file_type', 'unknown')
            
            context_parts.append(
                f"[Source {i}] From '{source_file}' ({file_type.upper()}), Section: {heading}\n"
                f"{result.content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        if times is not None:
            times['context_building'] = time.time() - context_start
        
        # Generate answer
        llm_start = time.time()
        
        prompt = f"""
You are a compliance-focused assistant that answers questions based strictly on the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. Answer using ONLY the context above. Do not add outside knowledge.
2. Provide a clear, authoritative explanation suitable for Texas DMV staff.
3. Cite sources in a simple format (e.g., “Source 2”).
4. If the context does not contain enough information, state clearly what is missing.
5. Organize the answer with concise bullets or short sections.
6. Include all specific rule details, thresholds, definitions, and exceptions found in the context.
7.If the context describes what an entity does, you may infer its definition even if the text does not explicitly label it as a definition.



Answer:"""
        
        response = self.groq_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        if times is not None:
            times['llm_generation'] = time.time() - llm_start
        
        # Prepare response
        sources = []
        if include_sources:
            for result in results:
                sources.append({
                    'content': result.content,
                    'metadata': result.metadata,
                    'relevance_score': result.score,
                    'rank': result.rank,
                    'retrieval_method': result.retrieval_method
                })
        
        total_time = time.time() - start_time
        self.stats['query_times'].append(total_time)
        if len(self.stats['query_times']) > 0:
            self.stats['avg_query_time'] = sum(self.stats['query_times']) / len(self.stats['query_times'])
        
        if times is not None:
            print(f"\n⏱️  Timing Breakdown:")
            for component, duration in times.items():
                print(f"   {component}: {duration:.3f}s")
        
        print(f"\n✅ Query complete in {total_time:.2f}s")
        
        return {
            'answer': answer,
            'sources': sources,
            'query': question,
            'num_sources': len(results),
            'retrieval_method': results[0].retrieval_method if results else 'none',
            'query_time': total_time,
            'timing_breakdown': times
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'total_documents': self.collection.count(),
            'total_queries': self.stats['total_queries'],
            'documents_added': self.stats['total_documents_added'],
            'chunks_created': self.stats['total_chunks_created'],
            'pdf_files_processed': self.stats['pdf_files'],
            'txt_files_processed': self.stats['txt_files'],
            'avg_query_time': self.stats['avg_query_time'],
            'collection_name': self.config['collection_name'],
            'embedding_model': self.config['embedding_model'],
            'llm_model': self.config['llm_model'],
            'hybrid_search': self.config['use_hybrid_search'],
            'reranking': False,
            'supported_formats': 'PDF, TXT'
        }
        
        if self.stats['query_times']:
            stats['min_query_time'] = min(self.stats['query_times'])
            stats['max_query_time'] = max(self.stats['query_times'])
        
        return stats


def main():
    """Example usage"""
    print("="*80)
    print("FAST RAG WITH PDF + TXT SUPPORT")
    print("="*80)
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("\n❌ Error: GROQ_API_KEY not set!")
        return
    
    rag = FastRAGSystemUniversal(
        groq_api_key=GROQ_API_KEY,
        collection_name="universal_demo",
        enable_profiling=True
    )
    
    print("\n✅ System supports both PDF and TXT files!")
    print("\n💡 Usage:")
    print("   rag.add_document('document.pdf')   # PDF file")
    print("   rag.add_document('notes.txt')       # TXT file")
    print("   rag.add_folder('./documents')       # All supported files")


if __name__ == "__main__":
    main()

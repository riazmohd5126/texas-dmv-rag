"""
Complete Advanced RAG Application
==================================

This is a production-ready RAG system that combines:
1. Structure-aware chunking (Stage 1, 2, 3)
2. Hybrid retrieval (dense + sparse)
3. Re-ranking for improved quality
4. Rich metadata tracking
5. Query optimization

Perfect for regulatory documents, manuals, and knowledge bases.
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Import our custom modules
from pdf_extractor import PDFStructureExtractor
from chunking_structure_aware import StructureAwareSplitter
from advanced_retrieval import HybridRetriever, RetrievalResult


class AdvancedRAGSystem:
    """
    Complete RAG system with advanced features.
    
    Features:
        ✅ Structure-aware chunking
        ✅ Hybrid retrieval (dense + sparse)
        ✅ Re-ranking with cross-encoder
        ✅ Rich metadata
        ✅ Query expansion
        ✅ Source attribution
    """
    
    def __init__(
        self,
        groq_api_key: str,
        collection_name: str = "advanced_rag",
        persist_directory: str = "./chroma_advanced",
        embedding_model: str = "all-mpnet-base-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        use_hybrid_search: bool = True,
        use_reranking: bool = True
    ):
        """
        Initialize the advanced RAG system.
        
        Args:
            groq_api_key: Your Groq API key
            collection_name: Name for the ChromaDB collection
            persist_directory: Where to store the database
            embedding_model: SentenceTransformer model name
            llm_model: Groq LLM model to use
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks
            use_hybrid_search: Enable dense + sparse retrieval
            use_reranking: Enable re-ranking for better results
        """
        print("\n" + "="*80)
        print("INITIALIZING ADVANCED RAG SYSTEM")
        print("="*80)
        
        # Store configuration
        self.config = {
            'collection_name': collection_name,
            'embedding_model': embedding_model,
            'llm_model': llm_model,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'use_hybrid_search': use_hybrid_search,
            'use_reranking': use_reranking
        }
        
        # Initialize Groq client
        print("\n🤖 Initializing Groq LLM...")
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model = llm_model
        print(f"   Model: {llm_model}")
        
        # Initialize embedding model
        print(f"\n🧠 Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"   Dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Initialize ChromaDB
        print(f"\n💾 Initializing ChromaDB...")
        print(f"   Location: {persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"   ✅ Loaded existing collection: {collection_name}")
            print(f"   Documents: {self.collection.count()}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"   ✅ Created new collection: {collection_name}")
        
        # Initialize PDF extractor
        print("\n📄 Initializing PDF extractor...")
        self.pdf_extractor = PDFStructureExtractor(preserve_structure=True)
        
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
            print("   Mode: Hybrid (dense + sparse)")
            self.retriever = HybridRetriever(
                collection=self.collection,
                embedding_model=self.embedding_model,
                use_reranking=use_reranking,
                dense_weight=0.7,
                sparse_weight=0.3
            )
        else:
            print("   Mode: Dense only (semantic search)")
            self.retriever = None
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'total_documents_added': 0,
            'total_chunks_created': 0
        }
        
        print("\n" + "="*80)
        print("✅ INITIALIZATION COMPLETE")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"   Collection: {collection_name}")
        print(f"   Documents: {self.collection.count()}")
        print(f"   Chunk size: {chunk_size} tokens (~{chunk_size*4} chars)")
        print(f"   Overlap: {chunk_overlap} tokens (~{chunk_overlap*4} chars)")
        print(f"   Hybrid search: {'✅ Enabled' if use_hybrid_search else '❌ Disabled'}")
        print(f"   Re-ranking: {'✅ Enabled' if use_reranking else '❌ Disabled'}")
        print("="*80 + "\n")
    
    def add_document(
        self,
        pdf_path: str,
        additional_metadata: Optional[Dict] = None
    ) -> int:
        """
        Add a PDF document to the RAG system.
        
        Process:
            1. Extract text with structure preservation
            2. Split using structure-aware chunking
            3. Generate embeddings
            4. Add to ChromaDB with metadata
            5. Index for sparse retrieval (if hybrid enabled)
        
        Args:
            pdf_path (str): Path to PDF file
            additional_metadata (dict): Optional extra metadata
            
        Returns:
            Number of chunks created from this document
        """
        print(f"\n{'='*80}")
        print(f"ADDING DOCUMENT: {Path(pdf_path).name}")
        print(f"{'='*80}")
        
        # Step 1: Extract PDF
        print("\n📄 Step 1: Extracting PDF...")
        extraction_result = self.pdf_extractor.extract_from_file(pdf_path)
        
        # Step 2: Prepare metadata
        base_metadata = {
            'source_file': Path(pdf_path).name,
            'source_path': str(Path(pdf_path).absolute()),
            'file_type': 'pdf',
            'total_pages': extraction_result['page_count'],
        }
        
        if extraction_result['metadata']:
            base_metadata.update({
                f'pdf_{k}': v for k, v in extraction_result['metadata'].items()
            })
        
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        # Step 3: Split into chunks
        print("\n🔪 Step 2: Structure-aware chunking...")
        chunks = self.chunker.split(
            text=extraction_result['text'],
            metadata=base_metadata
        )
        
        # Step 4: Prepare for ChromaDB
        print("\n💾 Step 3: Adding to database...")
        
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(pdf_path).stem}_chunk_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        print("   Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Add to ChromaDB
        print("   Adding to ChromaDB...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        # Step 5: Index for sparse retrieval
        if self.config['use_hybrid_search']:
            print("\n🔍 Step 4: Indexing for sparse retrieval...")
            self.retriever.index_for_sparse()
        
        # Update statistics
        self.stats['total_documents_added'] += 1
        self.stats['total_chunks_created'] += len(chunks)
        
        print(f"\n✅ Successfully added document!")
        print(f"   Created: {len(chunks)} chunks")
        print(f"   Total in database: {self.collection.count()}")
        print(f"{'='*80}\n")
        
        return len(chunks)
    
    def add_folder(
        self,
        folder_path: str,
        file_pattern: str = "*.pdf",
        recursive: bool = True,
        additional_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add all PDFs from a folder.
        
        Args:
            folder_path (str): Path to folder
            file_pattern (str): File pattern to match
            recursive (bool): Search subdirectories
            additional_metadata (dict): Metadata for all documents
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING FOLDER: {folder_path}")
        print(f"{'='*80}\n")
        
        # Find files
        folder = Path(folder_path)
        if recursive:
            files = list(folder.rglob(file_pattern))
        else:
            files = list(folder.glob(file_pattern))
        
        print(f"📁 Found {len(files)} files matching '{file_pattern}'\n")
        
        if not files:
            print("⚠️  No files found!")
            return {'files_processed': 0, 'total_chunks': 0}
        
        # Process each file
        total_chunks = 0
        successful = 0
        failed = []
        
        for i, pdf_file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Processing {pdf_file.name}")
            
            try:
                chunks = self.add_document(str(pdf_file), additional_metadata)
                total_chunks += chunks
                successful += 1
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append((pdf_file.name, str(e)))
        
        # Final summary
        print(f"\n{'='*80}")
        print("FOLDER PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"\n📊 Results:")
        print(f"   Total files: {len(files)}")
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
            'failed_files': failed
        }
    
    def query(
        self,
        question: str,
        top_k: int = 10,
        include_sources: bool = True,
        metadata_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Query the RAG system.
        
        Process:
            1. Retrieve relevant chunks (hybrid if enabled)
            2. Build context from top results
            3. Generate answer using LLM
            4. Return answer with sources
        
        Args:
            question (str): User's question
            top_k (int): Number of chunks to retrieve
            include_sources (bool): Include source information
            metadata_filter (dict): Filter by metadata
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n{'='*80}")
        print(f"QUERY: {question}")
        print(f"{'='*80}")
        
        self.stats['total_queries'] += 1
        
        # Step 1: Retrieve relevant chunks
        print(f"\n🔍 Retrieving relevant information...")
        print(f"   Method: {'Hybrid + Re-ranking' if self.config['use_hybrid_search'] else 'Dense only'}")
        print(f"   Retrieving top {top_k} chunks...")
        
        if self.config['use_hybrid_search'] and self.retriever:
            # Use hybrid retrieval
            results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                where=metadata_filter
            )
        else:
            # Use dense retrieval only
            query_embedding = self.embedding_model.encode([question])[0]
            
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=metadata_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Convert to RetrievalResult objects
            results = []
            for i, (doc, meta, dist) in enumerate(zip(
                chroma_results['documents'][0],
                chroma_results['metadatas'][0],
                chroma_results['distances'][0]
            )):
                from advanced_retrieval import RetrievalResult
                results.append(RetrievalResult(
                    content=doc,
                    metadata=meta,
                    score=1 - dist,
                    retrieval_method='dense',
                    rank=i + 1
                ))
        
        if not results:
            print("   ⚠️  No relevant information found!")
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'query': question
            }
        
        print(f"\n   ✅ Retrieved {len(results)} chunks")
        print(f"   Relevance scores: {results[0].score:.3f} - {results[-1].score:.3f}")
        
        # Step 2: Build context
        print(f"\n💭 Building context from top chunks...")
        
        context_parts = []
        for i, result in enumerate(results, 1):
            # Format each chunk with metadata
            heading = result.metadata.get('heading', 'Unknown section')
            source_file = result.metadata.get('source_file', 'Unknown source')
            
            context_parts.append(
                f"[Source {i}] From '{source_file}', Section: {heading}\n"
                f"{result.content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        print(f"   Context length: {len(context)} characters")
        
        # Step 3: Generate answer
        print(f"\n🤖 Generating answer with {self.llm_model}...")
        
        # Create prompt
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context information from documents:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information in the context above
2. Be specific and cite which source(s) you're using (e.g., "According to Source 1...")
3. If the context doesn't contain enough information, say so clearly
4. Organize your answer with clear structure (bullet points, numbered lists) when appropriate
5. Include specific details like numbers, dates, requirements, etc.

Answer:"""
        
        # Call Groq API
        response = self.groq_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Low temperature for factual answers
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        print(f"   ✅ Answer generated ({len(answer)} characters)")
        
        # Step 4: Prepare response with sources
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
        
        print(f"\n{'='*80}")
        print("✅ QUERY COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            'answer': answer,
            'sources': sources,
            'query': question,
            'num_sources': len(results),
            'retrieval_method': results[0].retrieval_method if results else 'none'
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_documents': self.collection.count(),
            'total_queries': self.stats['total_queries'],
            'documents_added': self.stats['total_documents_added'],
            'chunks_created': self.stats['total_chunks_created'],
            'collection_name': self.config['collection_name'],
            'embedding_model': self.config['embedding_model'],
            'llm_model': self.config['llm_model'],
            'hybrid_search': self.config['use_hybrid_search'],
            'reranking': self.config['use_reranking']
        }


def main():
    """
    Example usage of the complete RAG system.
    """
    print("="*80)
    print("COMPLETE ADVANCED RAG APPLICATION - EXAMPLE")
    print("="*80)
    
    # Check API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("\n❌ Error: GROQ_API_KEY not set!")
        print("Run: export GROQ_API_KEY='your-key-here'")
        return
    
    # Initialize the RAG system
    print("\n🚀 Initializing Advanced RAG System...")
    
    rag = AdvancedRAGSystem(
        groq_api_key=GROQ_API_KEY,
        collection_name="texas_dmv_complete",
        persist_directory="./chroma_complete",
        embedding_model="all-mpnet-base-v2",  # High quality
        chunk_size=700,                         # Good for regulatory docs
        chunk_overlap=150,                      # High overlap
        use_hybrid_search=True,                 # Enable hybrid retrieval
        use_reranking=True                      # Enable re-ranking
    )
    
    # Example 1: Add a document
    print("\n" + "="*80)
    print("EXAMPLE 1: Adding a document")
    print("="*80)
    
    # Uncomment and use your actual file:
    # rag.add_document(
    #     pdf_path="your_document.pdf",
    #     additional_metadata={
    #         'category': 'regulatory',
    #         'department': 'DMV'
    #     }
    # )
    
    # Example 2: Add a folder
    # print("\n" + "="*80)
    # print("EXAMPLE 2: Adding a folder of documents")
    # print("="*80)
    
    # rag.add_folder(
    #     folder_path="./documents",
    #     recursive=True,
    #     additional_metadata={'category': 'regulatory'}
    # )
    
    # Example 3: Query the system
    if rag.collection.count() > 0:
        print("\n" + "="*80)
        print("EXAMPLE 3: Querying the system")
        print("="*80)
        
        test_queries = [
            "What are the surety bond requirements for independent motor vehicle dealers?",
            "What documentation is needed for a license application?",
            "What are the office structure requirements?"
        ]
        
        for query in test_queries:
            response = rag.query(
                question=query,
                top_k=10,
                include_sources=True
            )
            
            print(f"\n{'='*80}")
            print(f"Question: {query}")
            print(f"{'='*80}")
            print(f"\n{response['answer']}")
            
            print(f"\n📚 Top 3 Sources:")
            for i, source in enumerate(response['sources'][:3], 1):
                print(f"\n{i}. Relevance: {source['relevance_score']:.3f}")
                print(f"   File: {source['metadata'].get('source_file', 'Unknown')}")
                print(f"   Section: {source['metadata'].get('heading', 'Unknown')}")
                print(f"   Preview: {source['content'][:150]}...")
    
    # Show statistics
    print("\n" + "="*80)
    print("SYSTEM STATISTICS")
    print("="*80)
    
    stats = rag.get_stats()
    print(f"\n📊 Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*80)
    print("✅ EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

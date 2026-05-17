"""
PDF Structure Extractor
=======================

This module extracts text from PDFs while preserving:
- Document structure (headings, sections)
- Formatting (lists, bullets, indentation)
- Metadata (page numbers, sections)

Optimized for regulatory documents, manuals, and structured content.
"""

import PyPDF2
from typing import List, Dict, Optional
from pathlib import Path
import re


class PDFStructureExtractor:
    """
    Extracts text from PDFs with structure preservation.
    
    This class handles:
    - Page-by-page extraction
    - Structure detection (headings, lists)
    - Format preservation
    - Metadata extraction
    """
    
    def __init__(self, preserve_structure: bool = True):
        """
        Initialize the PDF extractor.
        
        Args:
            preserve_structure (bool): Whether to preserve formatting
        """
        self.preserve_structure = preserve_structure
        print(f"✅ PDF Extractor initialized (preserve_structure={preserve_structure})")
    
    def extract_from_file(self, pdf_path: str) -> Dict:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dictionary with:
                - 'text': Full extracted text
                - 'pages': List of page texts
                - 'metadata': PDF metadata (title, author, etc.)
                - 'page_count': Number of pages
        
        Example:
            >>> extractor = PDFStructureExtractor()
            >>> result = extractor.extract_from_file("document.pdf")
            >>> print(f"Extracted {len(result['text'])} characters")
            >>> print(f"Pages: {result['page_count']}")
        """
        print(f"\n📄 Extracting from: {Path(pdf_path).name}")
        
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = self._extract_metadata(pdf_reader)
                page_count = len(pdf_reader.pages)
                
                print(f"   Pages: {page_count}")
                if metadata.get('title'):
                    print(f"   Title: {metadata['title']}")
                
                # Extract text from each page
                pages = []
                full_text = []
                
                for page_num in range(page_count):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # Clean up the page text
                    if self.preserve_structure:
                        page_text = self._preserve_structure(page_text, page_num + 1)
                    else:
                        page_text = self._basic_clean(page_text)
                    
                    pages.append(page_text)
                    full_text.append(page_text)
                
                # Combine all pages with page markers
                combined_text = '\n\n'.join(full_text)
                
                print(f"   ✅ Extracted {len(combined_text)} characters")
                
                return {
                    'text': combined_text,
                    'pages': pages,
                    'metadata': metadata,
                    'page_count': page_count,
                    'file_path': pdf_path,
                    'file_name': Path(pdf_path).name
                }
        
        except Exception as e:
            print(f"   ❌ Error extracting PDF: {e}")
            raise
    
    def _extract_metadata(self, pdf_reader: PyPDF2.PdfReader) -> Dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_reader: PyPDF2 reader object
            
        Returns:
            Dictionary with metadata fields
        """
        metadata = {}
        
        try:
            if pdf_reader.metadata:
                # Extract common metadata fields
                metadata['title'] = pdf_reader.metadata.get('/Title', '')
                metadata['author'] = pdf_reader.metadata.get('/Author', '')
                metadata['subject'] = pdf_reader.metadata.get('/Subject', '')
                metadata['creator'] = pdf_reader.metadata.get('/Creator', '')
                
                # Clean up metadata (remove None values)
                metadata = {k: v for k, v in metadata.items() if v}
        
        except Exception as e:
            print(f"   ⚠️  Could not extract metadata: {e}")
        
        return metadata
    
    def _preserve_structure(self, text: str, page_num: int) -> str:
        """
        Clean and preserve structure in extracted text.
        
        Args:
            text (str): Raw extracted text from page
            page_num (int): Page number
            
        Returns:
            Cleaned text with structure preserved
            
        What this does:
            1. Removes excessive whitespace
            2. Preserves line breaks at structural points
            3. Keeps numbered lists and bullets
            4. Maintains paragraph separation
        """
        if not text:
            return ""
        
        # Step 1: Fix common PDF extraction issues
        # Remove hyphenation at line breaks (e.g., "exam-\nple" → "example")
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Step 2: Normalize whitespace but preserve structure
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 3: Preserve important line breaks
        lines = text.split('\n')
        preserved_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check if this line should start a new paragraph
            starts_new_paragraph = False
            
            # Numbered items (1., 2., 1.1, etc.)
            if re.match(r'^\d+\.(\d+\.)*\s', line):
                starts_new_paragraph = True
            
            # Bullet points (•, -, *)
            if re.match(r'^[•\-\*]\s', line):
                starts_new_paragraph = True
            
            # Lines that look like headings (all caps, or numbered heading)
            if re.match(r'^[A-Z][A-Z\s]{10,}$', line):  # ALL CAPS
                starts_new_paragraph = True
            
            # If previous line ended with period, question, or exclamation
            if preserved_lines and preserved_lines[-1].endswith(('.', '?', '!')):
                # Check if current line starts with capital letter
                if line[0].isupper():
                    starts_new_paragraph = True
            
            # Add appropriate separation
            if starts_new_paragraph and preserved_lines:
                preserved_lines.append('')  # Add blank line
            
            preserved_lines.append(line)
        
        # Join lines back together
        cleaned_text = '\n'.join(preserved_lines)
        
        # Step 4: Add page number reference
        # This helps track where information came from
        cleaned_text = f"[Page {page_num}]\n{cleaned_text}"
        
        return cleaned_text
    
    def _basic_clean(self, text: str) -> str:
        """
        Basic text cleaning without structure preservation.
        
        Args:
            text (str): Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_with_pages_separate(self, pdf_path: str) -> List[Dict]:
        """
        Extract text keeping each page as a separate chunk.
        
        This is useful if you want to process pages individually.
        
        Args:
            pdf_path (str): Path to PDF
            
        Returns:
            List of dictionaries, one per page
        
        Example:
            >>> extractor = PDFStructureExtractor()
            >>> pages = extractor.extract_with_pages_separate("doc.pdf")
            >>> for page_data in pages[:3]:  # First 3 pages
            ...     print(f"Page {page_data['page_num']}: {len(page_data['text'])} chars")
        """
        result = self.extract_from_file(pdf_path)
        
        pages_data = []
        for i, page_text in enumerate(result['pages'], 1):
            pages_data.append({
                'page_num': i,
                'text': page_text,
                'metadata': {
                    **result['metadata'],
                    'page': i,
                    'total_pages': result['page_count'],
                    'file_name': result['file_name']
                }
            })
        
        return pages_data


def example_usage():
    """
    Example of how to use the PDFStructureExtractor.
    """
    print("="*80)
    print("PDF STRUCTURE EXTRACTOR - EXAMPLE")
    print("="*80)
    
    # Note: This example assumes you have a PDF file
    # Replace with your actual PDF path
    pdf_path = "/mnt/user-data/uploads/1764568544451_eLICENSING-UserGuide_Independent-GDN-Licensees.pdf"
    
    # Create extractor
    extractor = PDFStructureExtractor(preserve_structure=True)
    
    # Extract full document
    result = extractor.extract_from_file(pdf_path)
    
    # Show results
    print("\n" + "="*80)
    print("EXTRACTION RESULTS")
    print("="*80)
    print(f"\n📊 Statistics:")
    print(f"   Total pages: {result['page_count']}")
    print(f"   Total characters: {len(result['text']):,}")
    print(f"   Average per page: {len(result['text']) // result['page_count']:,}")
    
    if result['metadata']:
        print(f"\n📄 Metadata:")
        for key, value in result['metadata'].items():
            if value:
                print(f"   {key}: {value}")
    
    # Show first page preview
    if result['pages']:
        print(f"\n📖 First Page Preview:")
        print("-" * 80)
        preview = result['pages'][0][:500]
        print(preview + "..." if len(result['pages'][0]) > 500 else preview)
    
    # Also demonstrate page-by-page extraction
    print("\n" + "="*80)
    print("PAGE-BY-PAGE EXTRACTION")
    print("="*80)
    
    pages = extractor.extract_with_pages_separate(pdf_path)
    
    print(f"\n📑 Extracted {len(pages)} pages individually")
    print("\nFirst 3 pages:")
    for page_data in pages[:3]:
        print(f"\n   Page {page_data['page_num']}: {len(page_data['text'])} characters")
        print(f"   Preview: {page_data['text'][:100]}...")


if __name__ == "__main__":
    # Run example if this file is executed directly
    example_usage()

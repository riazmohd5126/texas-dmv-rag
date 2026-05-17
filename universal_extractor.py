"""
Universal Document Extractor
=============================

Extracts text from BOTH:
- PDF files (.pdf)
- Text files (.txt)

With structure preservation for both formats.
"""

import PyPDF2
from typing import Dict, Optional
from pathlib import Path
import re


class UniversalDocumentExtractor:
    """
    Extracts text from PDFs and TXT files with structure preservation.
    
    Supported formats:
    - .pdf (using PyPDF2)
    - .txt (plain text)
    
    Features:
    - Automatic format detection
    - Structure preservation
    - Metadata extraction
    """
    
    def __init__(self, preserve_structure: bool = True):
        """
        Initialize the universal extractor.
        
        Args:
            preserve_structure (bool): Whether to preserve formatting
        """
        self.preserve_structure = preserve_structure
        print(f"✅ Universal Document Extractor initialized")
        print(f"   Supported formats: PDF, TXT")
        print(f"   Structure preservation: {preserve_structure}")
    
    def extract_from_file(self, file_path: str) -> Dict:
        """
        Extract text from a file (PDF or TXT).
        
        Automatically detects file type and uses appropriate extractor.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            Dictionary with:
                - 'text': Full extracted text
                - 'pages': List of page/section texts
                - 'page_count': Number of pages/sections
                - 'metadata': File metadata
                - 'file_type': 'pdf' or 'txt'
        
        Example:
            >>> extractor = UniversalDocumentExtractor()
            >>> result = extractor.extract_from_file("document.pdf")
            >>> print(result['text'][:100])
            >>> print(f"Pages: {result['page_count']}")
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect file type
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            print(f"   Detected: PDF file")
            return self._extract_pdf(file_path)
        elif file_extension == '.txt':
            print(f"   Detected: TXT file")
            return self._extract_txt(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {file_extension}\n"
                f"Supported: .pdf, .txt"
            )
    
    def _extract_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extraction result dictionary
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                print(f"   📄 PDF: {num_pages} pages")
                
                pages = []
                full_text = []
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if self.preserve_structure:
                        page_text = self._clean_and_preserve_structure(page_text)
                    
                    pages.append(page_text)
                    full_text.append(page_text)
                
                # Get metadata
                metadata = {}
                if pdf_reader.metadata:
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', '')
                    }
                
                combined_text = '\n\n'.join(full_text)
                
                print(f"   ✅ Extracted: {len(combined_text):,} characters")
                
                return {
                    'text': combined_text,
                    'pages': pages,
                    'page_count': num_pages,
                    'metadata': metadata,
                    'file_type': 'pdf'
                }
        
        except Exception as e:
            raise Exception(f"Error extracting PDF {pdf_path.name}: {e}")
    
    def _extract_txt(self, txt_path: Path) -> Dict:
        """
        Extract text from a TXT file.
        
        Args:
            txt_path: Path to TXT file
            
        Returns:
            Extraction result dictionary
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            text = None
            encoding_used = None
            
            for encoding in encodings:
                try:
                    with open(txt_path, 'r', encoding=encoding) as file:
                        text = file.read()
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise Exception("Could not decode file with any supported encoding")
            
            print(f"   📝 TXT: {len(text):,} characters (encoding: {encoding_used})")
            
            # Split into "pages" (sections) by double newlines or major headings
            pages = self._split_txt_into_sections(text)
            
            if self.preserve_structure:
                text = self._clean_and_preserve_structure(text)
                pages = [self._clean_and_preserve_structure(p) for p in pages]
            
            print(f"   ✅ Extracted: {len(pages)} sections")
            
            return {
                'text': text,
                'pages': pages,
                'page_count': len(pages),
                'metadata': {
                    'filename': txt_path.name,
                    'encoding': encoding_used,
                    'size_bytes': txt_path.stat().st_size
                },
                'file_type': 'txt'
            }
        
        except Exception as e:
            raise Exception(f"Error extracting TXT {txt_path.name}: {e}")
    
    def _split_txt_into_sections(self, text: str) -> list:
        """
        Split a TXT file into logical sections.
        
        Sections are split by:
        1. Multiple blank lines (paragraph breaks)
        2. Lines that look like headings (numbered sections)
        
        Args:
            text: Full text content
            
        Returns:
            List of section texts
        """
        # Split by multiple newlines (2+)
        sections = re.split(r'\n\s*\n\s*\n+', text)
        
        # Filter out empty sections
        sections = [s.strip() for s in sections if s.strip()]
        
        # If we got too few sections, try splitting by numbered headings
        if len(sections) < 3:
            # Try to split by lines that look like headings
            # E.g., "1. Introduction", "Section 1:", "CHAPTER 1"
            heading_pattern = r'(?:^|\n)(?:\d+\.|\d+\)|\w+\s+\d+:?|\d+\.\d+)\s+[A-Z]'
            parts = re.split(heading_pattern, text)
            if len(parts) > len(sections):
                sections = [p.strip() for p in parts if p.strip()]
        
        # If still just one section, split into ~5000 char chunks
        if len(sections) == 1:
            text_len = len(text)
            chunk_size = 5000
            sections = []
            for i in range(0, text_len, chunk_size):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    sections.append(chunk)
        
        return sections if sections else [text]
    
    def _clean_and_preserve_structure(self, text: str) -> str:
        """
        Clean text while preserving structure.
        
        This:
        - Fixes spacing issues
        - Preserves headings
        - Preserves lists and bullets
        - Removes excessive whitespace
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with preserved structure
        """
        if not text:
            return ""
        
        # Remove multiple spaces (but keep newlines)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])-\s+([a-z])', r'\1\2', text)  # Fix hyphenated words
        
        # Preserve numbered lists
        text = re.sub(r'(\d+\.)\s+', r'\n\1 ', text)
        
        # Preserve bullet points
        text = re.sub(r'([•●○▪▫■□])\s+', r'\n\1 ', text)
        
        # Preserve section numbers (e.g., "1.1", "2.3.4")
        text = re.sub(r'(\d+(?:\.\d+)+)\s+', r'\n\1 ', text)
        
        # Remove excessive blank lines (keep max 2)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Clean up leading/trailing whitespace
        text = text.strip()
        
        return text


def example_usage():
    """Example of how to use the UniversalDocumentExtractor"""
    
    print("="*80)
    print("UNIVERSAL DOCUMENT EXTRACTOR - EXAMPLE")
    print("="*80)
    
    extractor = UniversalDocumentExtractor(preserve_structure=True)
    
    print("\n📚 This extractor supports:")
    print("   • PDF files (.pdf)")
    print("   • Text files (.txt)")
    print("\n💡 Usage:")
    print("""
# Initialize
extractor = UniversalDocumentExtractor()

# Extract from PDF
result = extractor.extract_from_file("document.pdf")

# Extract from TXT
result = extractor.extract_from_file("notes.txt")

# Both return the same format:
print(result['text'])           # Full text
print(result['page_count'])     # Number of pages/sections
print(result['file_type'])      # 'pdf' or 'txt'
print(result['metadata'])       # File metadata
    """)


if __name__ == "__main__":
    example_usage()

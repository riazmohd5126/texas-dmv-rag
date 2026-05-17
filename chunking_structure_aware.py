"""
Structure-Aware Text Splitter
==============================
Three-stage chunking: headings → size enforcement → format preservation.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    text: str
    metadata: Dict = field(default_factory=dict)
    chunk_id: int = 0


class HeadingDetector:
    """Detects section headings in text."""
    
    PATTERNS = [
        re.compile(r'^(\d+\.(?:\d+\.)*)\s+(.+)$', re.MULTILINE),   # 1.1 Heading
        re.compile(r'^([A-Z][A-Z\s]{10,})$', re.MULTILINE),         # ALL CAPS
        re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE),             # Markdown
    ]
    
    def find_headings(self, text: str) -> List[Dict]:
        headings = []
        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                headings.append({
                    'text': match.group(0).strip(),
                    'start': match.start(),
                    'end': match.end(),
                    'level': self._detect_level(match.group(0).strip())
                })
        headings.sort(key=lambda h: h['start'])
        return headings
    
    def _detect_level(self, heading: str) -> int:
        if re.match(r'^\d+\.\d+\.\d+', heading):
            return 3
        if re.match(r'^\d+\.\d+', heading):
            return 2
        return 1


class StructureAwareSplitter:
    """
    Three-stage text splitter:
    1. Split by headings
    2. Enforce max chunk size
    3. Preserve formatting (lists, bullets)
    """
    
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 150,
                 max_chunk_size: int = None, overlap: int = None,
                 respect_structure: bool = True, **kwargs):
        self.chunk_size = max_chunk_size or chunk_size
        self.chunk_overlap = overlap or chunk_overlap
        self.heading_detector = HeadingDetector()
        self.char_limit = self.chunk_size * 4
        self.respect_structure = respect_structure
    
    def _is_junk_chunk(self, text: str) -> bool:
        """
        Detect table-of-contents entries, page headers, and other noise.
        These pollute retrieval results — filter them at ingestion time.
        """
        # TOC lines: lots of dots followed by page numbers
        # e.g. "8.3 Closure Application ................................ 95"
        dot_count = text.count('.')
        dot_ratio = dot_count / max(len(text), 1)
        if dot_ratio > 0.10 and re.search(r'\.\s*\d{1,3}\s*$', text):
            return True
        
        # Lines that are mostly dots/periods (TOC filler)
        if dot_ratio > 0.15:
            return True
        
        # Figure/Table references with no real content
        # e.g. "Figure 37: Previous Held Licenses .............. 36"
        if re.match(r'^(Figure|Table)\s+\d+', text) and '..' in text:
            return True
        
        # Too short to be useful content (just a heading with no body)
        stripped = re.sub(r'[\d\.\s\-:]+', '', text)  # Remove numbers, dots, spaces
        if len(stripped) < 40:
            return True
        
        return False
    
    def split(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        base_meta = metadata or {}
        
        # Stage 1: Split by headings
        sections = self._split_by_headings(text)
        
        # Stage 2: Enforce size limits
        sized_chunks = []
        for section in sections:
            if len(section['text']) > self.char_limit:
                sub_chunks = self._enforce_size(section['text'])
                for i, sc in enumerate(sub_chunks):
                    sized_chunks.append({
                        'text': sc,
                        'heading': section.get('heading', ''),
                        'part': i + 1
                    })
            else:
                sized_chunks.append(section)
        
        # Stage 3: Build TextChunk objects with metadata
        result = []
        for i, chunk_data in enumerate(sized_chunks):
            chunk_text = chunk_data['text'].strip()
            if not chunk_text or len(chunk_text) < 50:
                continue
            
            # Skip TOC entries, figure references, and other noise
            if self._is_junk_chunk(chunk_text):
                continue
            
            chunk_meta = {
                **base_meta,
                'heading': chunk_data.get('heading', ''),
                'section_number': i,
                'chunk_index': i,
            }
            result.append(TextChunk(text=chunk_text, metadata=chunk_meta, chunk_id=i))
        
        return result
    
    def _split_by_headings(self, text: str) -> List[Dict]:
        headings = self.heading_detector.find_headings(text)
        
        if not headings:
            return [{'text': text, 'heading': ''}]
        
        sections = []
        for i, heading in enumerate(headings):
            start = heading['start']
            end = headings[i + 1]['start'] if i + 1 < len(headings) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append({
                    'text': section_text,
                    'heading': heading['text']
                })
        
        # Include any text before the first heading
        if headings[0]['start'] > 0:
            pre_text = text[:headings[0]['start']].strip()
            if pre_text:
                sections.insert(0, {'text': pre_text, 'heading': 'Introduction'})
        
        return sections
    
    def _enforce_size(self, text: str) -> List[str]:
        """Split text into chunks respecting size limits."""
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.char_limit and current:
                chunks.append('\n\n'.join(current))
                # Keep overlap
                overlap_text = current[-1] if current else ""
                current = [overlap_text] if len(overlap_text) < self.chunk_overlap * 4 else []
                current_len = sum(len(c) for c in current)
            current.append(para)
            current_len += para_len
        
        if current:
            chunks.append('\n\n'.join(current))
        
        return chunks if chunks else [text]

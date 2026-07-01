import re
import fitz
from typing import List, Dict, Any
from loguru import logger

class BhagavadGitaParser:
    """
    A custom parser designed specifically for the Bhagavad Gita PDF.
    It splits pages into semantic units (e.g., introduction sections, 
    individual verses, and their corresponding commentaries).
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        # Pattern to match verse references like (1.21) or (1.21) (1.22) or (2.16)
        self.verse_pattern = re.compile(r'\((\d+)\.(\d+)\)')
        # Pattern to identify chapter headers
        self.chapter_pattern = re.compile(r'Chapter\s+(\d+)', re.IGNORECASE)

    def clean_text(self, text: str) -> str:
        """
        Cleans header/footer noises and abnormal spacing from page text.
        """
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Filter out running page headers and footers
            if not stripped:
                continue
            if "The Bhagavad Gita" in stripped:
                continue
            # Filter out numbers-only lines if they look like running headers
            if stripped.isdigit() and len(stripped) < 4:
                continue
            # Remove lines that are just repeated non-standard characters (e.g. Devanagari font artifacts)
            # Sanskrit Devanagari text printed using special non-unicode fonts contains characters like , H, etc.
            # We can retain them if needed, but english translation and comments are the key text.
            cleaned_lines.append(line)
            
        return "\n".join(cleaned_lines)

    def parse(self) -> List[Dict[str, Any]]:
        """
        Parses the PDF and returns a list of document chunks.
        Each chunk is represented as:
        {
            "text": str,  # The actual content of the chunk
            "metadata": {
                "source": str,
                "page": int,
                "chapter": int or None,
                "verse": str or None,
                "type": str  # 'intro', 'chapter_intro', 'verse'
            }
        }
        """
        logger.info(f"Opening PDF file: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        chunks = []
        
        current_chapter = None
        
        # Phase 1: Scan pages and extract content
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            
            # Identify chapter from page text if present
            chapter_matches = self.chapter_pattern.findall(raw_text)
            if chapter_matches:
                # Take the first matched chapter number
                current_chapter = int(chapter_matches[0])

            cleaned_text = self.clean_text(raw_text)
            if not cleaned_text.strip():
                continue

            # Determine page type
            # Pages 0 to ~30 are introduction pages
            if page_num < 32:
                page_type = "intro"
                chunks.append({
                    "text": cleaned_text,
                    "metadata": {
                        "source": self.pdf_path,
                        "page": page_num + 1,
                        "chapter": None,
                        "verse": None,
                        "type": page_type
                    }
                })
                continue
                
            # Check for verses on this page
            verse_matches = self.verse_pattern.findall(cleaned_text)
            
            if not verse_matches:
                # If no verses on a chapter page, it's likely a chapter introduction or transition page
                chunks.append({
                    "text": cleaned_text,
                    "metadata": {
                        "source": self.pdf_path,
                        "page": page_num + 1,
                        "chapter": current_chapter,
                        "verse": None,
                        "type": "chapter_intro"
                    }
                })
            else:
                # We have one or more verses on this page.
                # Let's group page content by verses.
                # Find all positions of verse markers e.g. (2.16)
                markers = list(self.verse_pattern.finditer(cleaned_text))
                
                # Split text based on these markers
                last_idx = 0
                for idx, marker in enumerate(markers):
                    start, end = marker.span()
                    verse_str = marker.group(0) # e.g. "(2.16)"
                    chap_num = int(marker.group(1))
                    verse_num = int(marker.group(2))
                    
                    # The text before this verse marker belongs to this verse translation/setup.
                    # Or, if it's the first marker, the text before is part of the first verse of the page.
                    # Let's capture the slice of text:
                    # Usually, the translation starts before the marker and the comment starts after the marker.
                    # A robust heuristic: each chunk contains the text from the end of the previous marker
                    # up to the end of the current marker, PLUS any commentary that follows it until the next verse or end of page.
                    
                    # Let's extract the verse unit.
                    # To make it simple and keep verse + comments together, we can capture the text:
                    # from the end of the previous marker (or start of page) to the start of the next marker (or end of page).
                    
                    next_start = markers[idx+1].span()[0] if idx + 1 < len(markers) else len(cleaned_text)
                    verse_block = cleaned_text[last_idx:next_start].strip()
                    
                    # Update metadata
                    chunks.append({
                        "text": verse_block,
                        "metadata": {
                            "source": self.pdf_path,
                            "page": page_num + 1,
                            "chapter": chap_num,
                            "verse": f"{chap_num}.{verse_num}",
                            "type": "verse"
                        }
                    })
                    last_idx = next_start
        
        logger.info(f"Successfully parsed PDF into {len(chunks)} semantic chunks")
        return chunks

if __name__ == "__main__":
    # Test the parser
    import sys
    parser = BhagavadGitaParser("data/The Bhagavad Gita.pdf")
    res = parser.parse()
    print(f"Total chunks: {len(res)}")
    if res:
        print("Sample chunk:")
        print(res[100])

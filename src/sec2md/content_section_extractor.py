"""Content-based section extraction for SEC filings.

This module provides a simpler, more predictable approach to section detection
by working on rendered markdown content rather than HTML DOM structure.

Key principles:
1. Work on markdown output (not HTML)
2. Use exact pattern matching (not fuzzy/styling heuristics)
3. Clear confidence levels for each detection method
4. Single unified strategy (not layered fallbacks)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Literal

# Section title mappings (exact matches, lowercase)
# Maps known section titles to their ITEM identifiers
SECTION_TITLES: Dict[str, str] = {
    # ITEM 1 - Business
    "business": "ITEM 1",
    "business summary": "ITEM 1",
    "description of business": "ITEM 1",
    "description of the business": "ITEM 1",
    "nature of business": "ITEM 1",
    "our business": "ITEM 1",
    "fundamentals of our business": "ITEM 1",
    "general": "ITEM 1",
    # ITEM 1A - Risk Factors
    "risk factors": "ITEM 1A",
    # ITEM 1B - Unresolved Staff Comments
    "unresolved staff comments": "ITEM 1B",
    # ITEM 1C - Cybersecurity
    "cybersecurity": "ITEM 1C",
    # ITEM 2 - Properties
    "properties": "ITEM 2",
    # ITEM 3 - Legal Proceedings
    "legal proceedings": "ITEM 3",
    # ITEM 4 - Mine Safety
    "mine safety disclosures": "ITEM 4",
    # ITEM 5 - Market for Common Equity
    "market for registrant's common equity": "ITEM 5",
    "market for registrant's common equity, related stockholder matters and issuer purchases of equity securities": "ITEM 5",
    # ITEM 6 - Reserved/Selected Financial Data
    "[reserved]": "ITEM 6",
    "reserved": "ITEM 6",
    "selected financial data": "ITEM 6",
    # ITEM 7 - MD&A
    "management's discussion and analysis": "ITEM 7",
    "management's discussion and analysis of financial condition and results of operations": "ITEM 7",
    "md&a": "ITEM 7",
    # ITEM 7A - Market Risk
    "quantitative and qualitative disclosures about market risk": "ITEM 7A",
    "market risk": "ITEM 7A",
    "financing and market risk": "ITEM 7A",
    # ITEM 8 - Financial Statements
    "financial statements and supplementary data": "ITEM 8",
    "financial statements": "ITEM 8",
    # ITEM 9 - Changes with Accountants
    "changes in and disagreements with accountants on accounting and financial disclosure": "ITEM 9",
    "changes in and disagreements with accountants": "ITEM 9",
    # ITEM 9A - Controls and Procedures
    "controls and procedures": "ITEM 9A",
    # ITEM 9B - Other Information
    "other information": "ITEM 9B",
    # ITEM 9C - Foreign Jurisdictions
    "disclosure regarding foreign jurisdictions that prevent inspections": "ITEM 9C",
    # ITEM 10 - Directors and Officers
    "directors, executive officers and corporate governance": "ITEM 10",
    "directors and executive officers": "ITEM 10",
    # ITEM 11 - Executive Compensation
    "executive compensation": "ITEM 11",
    # ITEM 12 - Security Ownership
    "security ownership of certain beneficial owners and management": "ITEM 12",
    "security ownership of certain beneficial owners and management and related stockholder matters": "ITEM 12",
    # ITEM 13 - Certain Relationships
    "certain relationships and related transactions": "ITEM 13",
    "certain relationships and related transactions, and director independence": "ITEM 13",
    # ITEM 14 - Principal Accountant Fees
    "principal accountant fees and services": "ITEM 14",
    # ITEM 15 - Exhibits
    "exhibits and financial statement schedules": "ITEM 15",
    "exhibits, financial statement schedules": "ITEM 15",
    "exhibit and financial statement schedules": "ITEM 15",
    # ITEM 16 - Form 10-K Summary
    "form 10-k summary": "ITEM 16",
}

# Filing structure defines valid PART/ITEM combinations
FILING_STRUCTURES = {
    "10-K": {
        "PART I": ["ITEM 1", "ITEM 1A", "ITEM 1B", "ITEM 1C", "ITEM 2", "ITEM 3", "ITEM 4"],
        "PART II": [
            "ITEM 5",
            "ITEM 6",
            "ITEM 7",
            "ITEM 7A",
            "ITEM 8",
            "ITEM 9",
            "ITEM 9A",
            "ITEM 9B",
            "ITEM 9C",
        ],
        "PART III": ["ITEM 10", "ITEM 11", "ITEM 12", "ITEM 13", "ITEM 14"],
        "PART IV": ["ITEM 15", "ITEM 16"],
    },
    "10-Q": {
        "PART I": ["ITEM 1", "ITEM 2", "ITEM 3", "ITEM 4"],
        "PART II": ["ITEM 1", "ITEM 1A", "ITEM 2", "ITEM 3", "ITEM 4", "ITEM 5", "ITEM 6"],
    },
}

# Regex patterns
ITEM_PATTERN = re.compile(
    r"^\s*\*?\*?\s*(ITEM\s+\d{1,2}[A-Z]?)\.?\s*[:\-–—]?\s*(.*)$", re.IGNORECASE
)

PART_PATTERN = re.compile(r"^\s*\*?\*?\s*(PART\s+[IVXLC]+)\s*$", re.IGNORECASE)

# TOC table pattern: | Item 1A. | Title | Page 31 |
TOC_TABLE_PATTERN = re.compile(
    r"\|\s*(Item\s+\d{1,2}[A-Z]?)\.?\s*\|\s*([^|]+)\|\s*(?:Pages?\s*)?(\d+)",
    re.IGNORECASE,
)


@dataclass
class SectionHeader:
    """A detected section header in the document."""

    line_num: int  # Line number in markdown
    item: str  # e.g., "ITEM 1A"
    title: str  # e.g., "Risk Factors"
    confidence: float  # 0.0 to 1.0
    part: Optional[str] = None  # e.g., "PART I"


class ContentBasedSectionExtractor:
    """Extract sections from SEC filings using content-based detection.

    This extractor works on rendered markdown content, using pattern matching
    and structural analysis rather than HTML DOM traversal.

    Detection strategies (in order of confidence):
    1. Explicit ITEM patterns: "ITEM 1A. Risk Factors" (confidence=1.0)
    2. Formatted title match: **RISK FACTORS** or RISK FACTORS (confidence=0.9)
    3. Structural title match: "Risk Factors" followed by blank then paragraph (confidence=0.7)
    """

    def __init__(
        self,
        filing_type: Optional[Literal["10-K", "10-Q", "20-F", "8-K"]] = None,
        debug: bool = False,
    ):
        self.filing_type = filing_type
        self.structure = FILING_STRUCTURES.get(filing_type) if filing_type else None
        self.debug = debug

    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def _infer_part_for_item(self, item: str) -> Optional[str]:
        """Infer PART from ITEM number based on filing structure."""
        if not self.filing_type or self.filing_type not in FILING_STRUCTURES:
            return None

        structure = FILING_STRUCTURES[self.filing_type]
        for part, items in structure.items():
            if item in items:
                return part
        return None

    def _parse_toc_entries(self, markdown: str) -> Dict[str, str]:
        """Parse TOC table entries to get ITEM -> title mappings.

        Returns dict mapping item (e.g., "ITEM 1A") to title (e.g., "Risk Factors")
        """
        entries = {}
        for match in TOC_TABLE_PATTERN.finditer(markdown):
            item = match.group(1).upper()
            item = re.sub(r"\s+", " ", item)  # Normalize spacing
            if not item.startswith("ITEM"):
                item = "ITEM " + item.replace("ITEM", "").strip()
            title = match.group(2).strip()
            entries[item] = title
            self._log(f"TOC entry: {item} -> {title}")
        return entries

    def _find_toc_section_headers(
        self, markdown: str, toc_entries: Dict[str, str]
    ) -> List[SectionHeader]:
        """Find section headers in content that match TOC entries.

        For each TOC entry, search for the title appearing as a standalone line
        in the document (after the TOC area).
        """
        headers = []
        lines = markdown.split("\n")

        # Skip TOC area (first ~10% of document or until we see "PART I" content)
        toc_end_line = len(lines) // 10  # Default: skip first 10%

        for item, title in toc_entries.items():
            title_lower = title.lower().strip()
            title_clean = re.sub(r"[^\w\s]", "", title_lower)  # Remove punctuation

            # Search for this title in the content
            for i in range(toc_end_line, len(lines)):
                line = lines[i].strip()
                line_lower = line.lower()
                line_clean = re.sub(r"[^\w\s]", "", line_lower)

                # Skip table rows (they're in TOC)
                if "|" in line:
                    continue

                # Match if line is substantially similar to title
                if line_clean == title_clean or title_clean in line_clean:
                    # Verify it looks like a header (short, standalone)
                    if len(line) < 100:
                        part = self._infer_part_for_item(item)
                        headers.append(
                            SectionHeader(
                                line_num=i,
                                item=item,
                                title=title,
                                confidence=0.6,  # TOC-matched = medium confidence
                                part=part,
                            )
                        )
                        self._log(f"Line {i}: TOC match -> {item} '{title}' (conf=0.6)")
                        break  # Found this item, move to next

        return headers

    def _is_header_by_structure(self, lines: List[str], i: int) -> bool:
        """Check if line i is a header based on structural patterns.

        A structural header is:
        - A short line (< 60 chars)
        - Followed by a blank line
        - Then followed by content (any non-empty text)

        We also look a bit further ahead since some headers are followed by
        a subsection header before the main content.
        """
        if i >= len(lines) - 2:
            return False

        current = lines[i].strip()
        next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

        is_short = 3 < len(current) < 60
        is_followed_by_blank = next_line == ""

        if not (is_short and is_followed_by_blank):
            return False

        # Look ahead for substantial content (within next 5 lines)
        for j in range(i + 2, min(i + 7, len(lines))):
            line = lines[j].strip()
            if len(line) > 80:
                return True

        return False

    def _clean_line(self, line: str) -> str:
        """Remove markdown formatting from line."""
        # Remove bold markers
        clean = line.strip().replace("**", "").replace("*", "")
        # Remove heading markers
        clean = re.sub(r"^#{1,6}\s*", "", clean)
        return clean.strip()

    def detect_headers(self, markdown: str) -> List[SectionHeader]:
        """Detect section headers in markdown content.

        Returns list of SectionHeader objects sorted by line number.
        """
        headers: List[SectionHeader] = []
        lines = markdown.split("\n")

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Skip empty lines, very long lines, table rows
            if not line_stripped or len(line_stripped) > 120 or "|" in line_stripped:
                continue

            # Strategy 1: Explicit ITEM pattern (highest confidence)
            match = ITEM_PATTERN.match(line_stripped)
            if match:
                item = match.group(1).upper()
                # Normalize spacing: "ITEM  1A" -> "ITEM 1A"
                item = re.sub(r"\s+", " ", item)
                title = self._clean_line(match.group(2))
                part = self._infer_part_for_item(item)

                headers.append(
                    SectionHeader(
                        line_num=i,
                        item=item,
                        title=title,
                        confidence=1.0,
                        part=part,
                    )
                )
                self._log(f"Line {i}: ITEM pattern -> {item} '{title}' (conf=1.0)")
                continue

            # Strategy 2 & 3: Title matching
            clean_line = self._clean_line(line_stripped)
            clean_lower = clean_line.lower()

            # Check formatting
            is_bold = line_stripped.startswith("**") or line_stripped.startswith("##")
            is_caps = clean_line.isupper() and len(clean_line) > 3

            # Look up in title dictionary
            if clean_lower in SECTION_TITLES:
                item = SECTION_TITLES[clean_lower]
                part = self._infer_part_for_item(item)

                if is_bold or is_caps:
                    # Strategy 2: Formatted title (high confidence)
                    headers.append(
                        SectionHeader(
                            line_num=i,
                            item=item,
                            title=clean_line,
                            confidence=0.9,
                            part=part,
                        )
                    )
                    self._log(f"Line {i}: Formatted title -> {item} '{clean_line}' (conf=0.9)")

                elif self._is_header_by_structure(lines, i):
                    # Strategy 3: Structural pattern (medium confidence)
                    headers.append(
                        SectionHeader(
                            line_num=i,
                            item=item,
                            title=clean_line,
                            confidence=0.7,
                            part=part,
                        )
                    )
                    self._log(f"Line {i}: Structural title -> {item} '{clean_line}' (conf=0.7)")

        # Strategy 4: TOC-based detection (for filings without formatted headers)
        # Always try TOC to fill in missing sections
        if True:  # Always check TOC
            toc_entries = self._parse_toc_entries(markdown)
            if toc_entries:
                self._log(f"Found {len(toc_entries)} TOC entries, using as fallback")
                toc_headers = self._find_toc_section_headers(markdown, toc_entries)
                # Merge TOC headers (only add items not already found)
                existing_items = {h.item for h in headers}
                for th in toc_headers:
                    if th.item not in existing_items:
                        headers.append(th)
                        self._log(f"Added {th.item} from TOC matching")

        return headers

    def _deduplicate_headers(self, headers: List[SectionHeader]) -> List[SectionHeader]:
        """Remove duplicate ITEM entries, keeping highest confidence."""
        # Group by item
        by_item: Dict[str, List[SectionHeader]] = {}
        for h in headers:
            if h.item not in by_item:
                by_item[h.item] = []
            by_item[h.item].append(h)

        # Keep best header for each item (highest confidence, then earliest)
        unique: List[SectionHeader] = []
        for item, item_headers in by_item.items():
            # Sort by confidence desc, then line_num asc
            item_headers.sort(key=lambda h: (-h.confidence, h.line_num))
            best = item_headers[0]
            unique.append(best)
            if len(item_headers) > 1:
                self._log(
                    f"Deduplicated {item}: kept line {best.line_num} "
                    f"(conf={best.confidence}), dropped {len(item_headers)-1} others"
                )

        # Sort by line number
        unique.sort(key=lambda h: h.line_num)
        return unique

    def extract_sections(self, pages: list) -> list:
        """Extract sections from parsed pages.

        Args:
            pages: List of Page objects from Parser

        Returns:
            List of Section objects
        """
        from sec2md.models import Section, Page

        # Combine pages into markdown with line tracking
        markdown_lines: List[str] = []
        line_to_page: Dict[int, int] = {}  # line_num -> page_num

        for page in pages:
            start_line = len(markdown_lines)
            page_lines = page.content.split("\n")
            markdown_lines.extend(page_lines)

            # Map each line to its page
            for j in range(len(page_lines)):
                line_to_page[start_line + j] = page.number

            # Add blank line between pages
            markdown_lines.append("")
            line_to_page[len(markdown_lines) - 1] = page.number

        markdown = "\n".join(markdown_lines)

        # Detect headers
        headers = self.detect_headers(markdown)
        headers = self._deduplicate_headers(headers)

        self._log(f"Found {len(headers)} unique section headers")

        if not headers:
            return []

        # Extract content between headers
        sections: List[Section] = []

        for i, header in enumerate(headers):
            # Determine content boundaries (line numbers)
            start_line = header.line_num
            if i + 1 < len(headers):
                end_line = headers[i + 1].line_num
            else:
                end_line = len(markdown_lines)

            # Extract content lines
            content_lines = markdown_lines[start_line:end_line]
            content = "\n".join(content_lines).strip()

            # Skip sections with minimal content
            if len(content) < 100:
                self._log(f"Skipping {header.item}: only {len(content)} chars")
                continue

            # Determine page range
            start_page = line_to_page.get(start_line, 1)
            end_page = line_to_page.get(end_line - 1, start_page)

            # Create pages for section (simplified: single page with all content)
            # In the future, could split by actual page boundaries
            section_page = Page(
                number=start_page,
                content=content,
                elements=None,
            )

            section = Section(
                part=header.part,
                item=header.item,
                item_title=header.title if header.title else None,
                pages=[section_page],
            )

            sections.append(section)
            self._log(
                f"Extracted {header.item}: {len(content)} chars, "
                f"pages {start_page}-{end_page}, conf={header.confidence}"
            )

        return sections

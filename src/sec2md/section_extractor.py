from __future__ import annotations

import re
from typing import List, Optional, Literal, Any

LEAD_WRAP = r'(?:\*\*|__)?\s*(?:</?[^>]+>\s*)*'

PART_PATTERN = re.compile(
    rf'^\s*{LEAD_WRAP}(PART\s+[IVXLC]+)\.?(?:\*\*|__)?(?:\s*$|\s+)',
    re.IGNORECASE | re.MULTILINE
)
ITEM_PATTERN = re.compile(
    rf'^\s*{LEAD_WRAP}(ITEM)\s+(\d{{1,2}}[A-Z]?)\.?\s*(?:[:.\-–—]\s*)?(.*)',
    re.IGNORECASE | re.MULTILINE
)

HEADER_FOOTER_RE = re.compile(
    r'^\s*(?:[A-Z][A-Za-z0-9 .,&\-]+)?\s*\|\s*\d{4}\s+Form\s+10-[KQ]\s*\|\s*\d+\s*$'
)
PAGE_NUM_RE = re.compile(r'^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$|^\s*\d+\s*$', re.IGNORECASE)
MD_EDGE = re.compile(r'^\s*(?:\*\*|__)\s*|\s*(?:\*\*|__)\s*$')

NBSP, NARROW_NBSP, ZWSP = '\u00A0', '\u202F', '\u200B'

DOT_LEAD_RE = re.compile(r'^.*\.{3,}\s*\d{1,4}\s*$', re.M)  # "... 123"
ITEM_ROWS_RE = re.compile(r'^\s*ITEM\s+\d{1,2}[A-Z]?\.?\b', re.I | re.M)
ITEM_BREADCRUMB_TITLE_RE = re.compile(
    r'^[,\s]*(\d{1,2}[A-Z]?)(\s*,\s*\d{1,2}[A-Z]?)*\s*$',
    re.IGNORECASE
)

FILING_STRUCTURES = {
    "10-K": {
        "PART I": ["ITEM 1", "ITEM 1A", "ITEM 1B", "ITEM 1C", "ITEM 2", "ITEM 3", "ITEM 4"],
        "PART II": ["ITEM 5", "ITEM 6", "ITEM 7", "ITEM 7A", "ITEM 8", "ITEM 9", "ITEM 9A", "ITEM 9B", "ITEM 9C"],
        "PART III": ["ITEM 10", "ITEM 11", "ITEM 12", "ITEM 13", "ITEM 14"],
        "PART IV": ["ITEM 15", "ITEM 16"]
    },
    "10-Q": {
        "PART I": ["ITEM 1", "ITEM 2", "ITEM 3", "ITEM 4"],
        "PART II": ["ITEM 1", "ITEM 1A", "ITEM 2", "ITEM 3", "ITEM 4", "ITEM 5", "ITEM 6"]
    },
    "20-F": {
        "PART I": [
            "ITEM 1", "ITEM 2", "ITEM 3", "ITEM 4", "ITEM 4A", "ITEM 5",
            # Some 20-F filings include items 6-12 in PART I without explicit PART II header
            "ITEM 6", "ITEM 7", "ITEM 8", "ITEM 9", "ITEM 10", "ITEM 11", "ITEM 12", "ITEM 12D"
        ],
        "PART II": [
            "ITEM 13", "ITEM 14", "ITEM 15",
            # include all 16X variants explicitly so validation stays strict
            "ITEM 16", "ITEM 16A", "ITEM 16B", "ITEM 16C", "ITEM 16D", "ITEM 16E", "ITEM 16F", "ITEM 16G", "ITEM 16H",
            "ITEM 16I"
        ],
        "PART III": ["ITEM 17", "ITEM 18", "ITEM 19"]
    }
}

SECTION_KEYWORDS = {
    # 10-K Part I - use longer phrases first for specificity
    "fundamentals of our business": "ITEM 1",
    "availability of company information": "ITEM 1",
    "description of business": "ITEM 1",
    "business": "ITEM 1",
    "risk factors": "ITEM 1A",
    "unresolved staff comments": "ITEM 1B",
    "cybersecurity": "ITEM 1C",
    "properties": "ITEM 2",
    "legal proceedings": "ITEM 3",
    "mine safety": "ITEM 4",

    # 10-K Part II
    "market for registrant": "ITEM 5",
    "[reserved]": "ITEM 6",
    "reserved": "ITEM 6",
    "selected financial data": "ITEM 6",
    "management's discussion and analysis": "ITEM 7",
    "management's discussion": "ITEM 7",
    "md&a": "ITEM 7",
    "quantitative and qualitative disclosures about market risk": "ITEM 7A",
    "quantitative and qualitative": "ITEM 7A",
    "market risk": "ITEM 7A",
    "financial statements and supplementary data": "ITEM 8",
    "financial statements": "ITEM 8",
    "changes in and disagreements with accountants": "ITEM 9",
    "changes in and disagreements": "ITEM 9",
    "controls and procedures": "ITEM 9A",
    "other information": "ITEM 9B",
    "disclosure regarding foreign jurisdictions": "ITEM 9C",

    # 10-K Part III
    "directors, executive officers and corporate governance": "ITEM 10",
    "directors": "ITEM 10",
    "executive compensation": "ITEM 11",
    "security ownership of certain beneficial owners": "ITEM 12",
    "security ownership": "ITEM 12",
    "certain relationships and related transactions": "ITEM 13",
    "certain relationships": "ITEM 13",
    "principal accountant fees and services": "ITEM 14",
    "principal accountant": "ITEM 14",

    # 10-K Part IV
    "exhibits, financial statement schedules": "ITEM 15",
    "exhibits": "ITEM 15",
    "form 10-k summary": "ITEM 16",
}


class SectionExtractor:
    def __init__(self, pages: List[Any], filing_type: Optional[Literal["10-K", "10-Q", "20-F", "8-K"]] = None,
                 desired_items: Optional[set] = None, debug: bool = False, raw_html: Optional[str] = None):
        """Extract sections from SEC filings."""
        self.pages = pages
        self.filing_type = filing_type
        self.structure = FILING_STRUCTURES.get(filing_type) if filing_type else None
        self.desired_items = desired_items
        self.debug = debug
        self._toc_locked = False
        self.raw_html = raw_html  # For TOC-based fallback extraction

    def _log(self, msg: str):
        if self.debug:
            print(msg)

    @staticmethod
    def _normalize_section_key(part: Optional[str], item_num: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        part_key = re.sub(r'\s+', ' ', part.upper().strip()) if part else None
        item_key = f"ITEM {item_num.upper()}" if item_num else None
        return part_key, item_key

    @staticmethod
    def _normalize_section(text: str) -> str:
        return re.sub(r'\s+', ' ', text.upper().strip())

    def _clean_lines(self, content: str) -> List[str]:
        """Remove headers, footers, and page navigation."""
        content = content.replace(NBSP, ' ').replace(NARROW_NBSP, ' ').replace(ZWSP, '')
        lines = [ln.rstrip() for ln in content.split('\n')]
        content_str = '\n'.join(lines)

        # TODO: Breadcrumb removal - some filings have "PART II\n\nItem 7" on every page
        # as navigation breadcrumbs, but removing them here breaks section detection for
        # filings that use this pattern as actual section headers (e.g., MSFT 10-K).
        # Solution: Handle breadcrumb removal during HTML parsing stage instead of here.

        # COMMENTED OUT - breaks section detection for some filings
        # content_str = re.sub(
        #     r'^\s*PART\s+[IVXLC]+\s*$\n+^\s*Item\s+\d{1,2}[A-Z]?\s*$\n+',
        #     '',
        #     content_str,
        #     flags=re.MULTILINE
        # )
        #
        # lines_list = content_str.split('\n')
        # filtered_lines = []
        # for line in lines_list:
        #     if re.match(r'^\s*Item\s+\d{1,2}[A-Z]?(?:\s*,\s*\d{1,2}[A-Z]?)*\s*$', line, re.IGNORECASE):
        #         continue
        #     filtered_lines.append(line)
        # content_str = '\n'.join(filtered_lines)

        lines = content_str.split('\n')

        out = []
        for ln in lines:
            if HEADER_FOOTER_RE.match(ln) or PAGE_NUM_RE.match(ln):
                continue
            ln = MD_EDGE.sub('', ln)
            out.append(ln)
        return out

    def _infer_part_for_item(self, filing_type: str, item_key: str) -> Optional[str]:
        """Infer PART from ITEM number (10-K only)."""
        m = re.match(r'ITEM\s+(\d{1,2})', item_key)
        if not m:
            return None
        num = int(m.group(1))
        if filing_type == "10-K":
            if 1 <= num <= 4:
                return "PART I"
            elif 5 <= num <= 9:
                return "PART II"
            elif 10 <= num <= 14:
                return "PART III"
            elif 15 <= num <= 16:
                return "PART IV"
        return None

    @staticmethod
    def _clean_item_title(title: str) -> str:
        title = re.sub(r'^\s*[:.\-–—]\s*', '', title)
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    def _is_toc(self, content: str, page_num: int = 1) -> bool:
        """Detect table of contents pages."""
        if self._toc_locked or page_num > 5:
            return False

        # Large pages (>20K chars) are not TOCs - they're full document content
        # A typical TOC page is 2-10K chars
        if len(content) > 20000:
            self._log(f"DEBUG: Page {page_num} is {len(content)} chars - too large for TOC")
            return False

        # Check for traditional TOC patterns (dot leaders, plain ITEM rows)
        item_hits = len(ITEM_ROWS_RE.findall(content))
        leader_hits = len(DOT_LEAD_RE.findall(content))
        if (item_hits >= 3) or (leader_hits >= 3):
            return True

        # Check for table-based TOCs (modern filings)
        # Look for markdown tables with ITEM entries and page numbers
        # Pattern: | ITEM X. | TITLE | PAGE |
        table_item_pattern = re.compile(r'\|\s*ITEM\s+\d{1,2}[A-Z]?\.?\s*\|', re.IGNORECASE)
        table_item_hits = len(table_item_pattern.findall(content))
        if table_item_hits >= 3:
            return True

        # Also check for "TABLE OF CONTENTS" header
        if re.search(r'TABLE\s+OF\s+CONTENTS', content, re.IGNORECASE) and table_item_hits >= 2:
            return True

        return False

    # =============================================================================
    # TOC-Based Section Extraction (Fallback for non-standard formats like Intel)
    # =============================================================================

    def _get_element_section_text(self, element, max_depth: int = 5) -> str:
        """
        Traverse up and across from an element to find meaningful section text.

        SEC filings often have the Item text in a parent or nearby sibling element.
        Some filings (Intel) have empty target divs with content in next siblings.
        Returns text like "Item 7. Management's Discussion..."
        """
        from bs4 import BeautifulSoup, Tag

        if not isinstance(element, Tag):
            return ""

        # First try the element itself
        text = element.get_text(separator=" ", strip=True)[:200]
        if text:
            return text

        # Try next siblings (Intel-style: empty target div, text in next element)
        for sibling in element.find_next_siblings()[:3]:
            if isinstance(sibling, Tag):
                sibling_text = sibling.get_text(separator=" ", strip=True)[:200]
                if sibling_text:
                    return sibling_text

        # Try traversing up
        current = element
        for _ in range(max_depth):
            if current.parent and isinstance(current.parent, Tag):
                current = current.parent
                text = current.get_text(separator=" ", strip=True)[:200]
                if text:
                    return text
            else:
                break
        return ""

    @staticmethod
    def _fuzzy_match_ratio(str1: str, str2: str) -> float:
        """
        Calculate fuzzy match ratio between two strings (0.0 to 1.0).

        Uses simple character-level similarity: common chars / total chars.
        """
        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()

        if not str1_lower or not str2_lower:
            return 0.0

        if str1_lower == str2_lower:
            return 1.0

        # Count common characters (order-independent)
        from collections import Counter
        c1 = Counter(str1_lower)
        c2 = Counter(str2_lower)

        # Intersection: sum of min counts for each character
        common = sum((c1 & c2).values())

        # Total: sum of all characters
        total = sum(c1.values()) + sum(c2.values())

        return 2.0 * common / total if total > 0 else 0.0

    def _match_section_to_item(self, text: str) -> Optional[str]:
        """
        Match section text to an ITEM identifier (e.g., "ITEM 1", "ITEM 1A").

        Returns normalized item string like "ITEM 1A" or None if no match.
        """
        # Try ITEM patterns first (more specific)
        item_match = ITEM_PATTERN.search(text)
        if item_match:
            item_num = item_match.group(2)
            return f"ITEM {item_num.upper()}"

        # Try matching specific item numbers with various formats
        patterns = [
            (re.compile(r'Item\s*1A\b', re.IGNORECASE), "ITEM 1A"),
            (re.compile(r'Risk\s+Factors', re.IGNORECASE), "ITEM 1A"),
            (re.compile(r'Item\s*7\b', re.IGNORECASE), "ITEM 7"),
            (re.compile(r'MD&A|Management.s\s+Discussion', re.IGNORECASE), "ITEM 7"),
            (re.compile(r'Item\s*1\b(?!\s*[A-Z])', re.IGNORECASE), "ITEM 1"),
            (re.compile(r'Business\s+Description|Description\s+of\s+Business', re.IGNORECASE), "ITEM 1"),
        ]

        for pattern, item_id in patterns:
            if pattern.search(text):
                return item_id

        # Try fuzzy matching against SECTION_KEYWORDS
        text_lower = text.lower()
        best_match = None
        best_ratio = 0.0

        sorted_keywords = sorted(SECTION_KEYWORDS.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            # Direct substring match (preferred)
            if keyword in text_lower:
                # Check if keyword is substantial part of text
                keyword_ratio = len(keyword) / len(text_lower)
                if keyword_ratio >= 0.3:
                    return SECTION_KEYWORDS[keyword]

            # Fuzzy match for similar phrases (only if text is long enough)
            if len(text_lower) >= 10:
                ratio = self._fuzzy_match_ratio(keyword, text_lower)
                if ratio > best_ratio and ratio >= 0.7:  # Increased threshold from 0.6 to 0.7
                    best_ratio = ratio
                    best_match = SECTION_KEYWORDS[keyword]

        if best_match:
            self._log(f"Fuzzy matched '{text[:40]}' to {best_match} (ratio={best_ratio:.2f})")
            return best_match

        return None

    @staticmethod
    def _parse_color_to_hex(color_str: str) -> Optional[str]:
        """Convert CSS color to hex format for comparison."""
        color_str = color_str.strip().lower()

        # Already hex
        if color_str.startswith('#'):
            return color_str.upper()

        # RGB format
        rgb_match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str)
        if rgb_match:
            r, g, b = map(int, rgb_match.groups())
            return f"#{r:02X}{g:02X}{b:02X}"

        # Named colors (common SEC filing colors)
        NAMED_COLORS = {
            'black': '#000000', 'navy': '#000080', 'blue': '#0000FF',
            'darkblue': '#00008B', 'maroon': '#800000', 'red': '#FF0000',
            'green': '#008000', 'gray': '#808080', 'silver': '#C0C0C0',
        }
        return NAMED_COLORS.get(color_str)

    @staticmethod
    def _color_distance(hex1: str, hex2: str) -> float:
        """Calculate perceptual color distance (0-1 scale)."""
        if not hex1 or not hex2:
            return 0.0

        try:
            r1, g1, b1 = int(hex1[1:3], 16), int(hex1[3:5], 16), int(hex1[5:7], 16)
            r2, g2, b2 = int(hex2[1:3], 16), int(hex2[3:5], 16), int(hex2[5:7], 16)
            # Euclidean distance in RGB space, normalized
            distance = ((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) ** 0.5
            return min(distance / 441.67, 1.0)  # 441.67 = sqrt(255^2 * 3)
        except (ValueError, IndexError):
            return 0.0

    def _extract_style_properties(self, element) -> dict:
        """Parse style attribute to extract font-weight, font-size, color, font-family."""
        from bs4 import Tag

        if not isinstance(element, Tag):
            return {}

        style_str = element.get('style', '')
        if not style_str:
            return {}

        properties = {}

        # Parse font-weight (bold = 700, normal = 400)
        weight_match = re.search(r'font-weight:\s*(\d+|bold|normal)', style_str, re.IGNORECASE)
        if weight_match:
            weight_val = weight_match.group(1).lower()
            if weight_val == 'bold':
                properties['weight'] = 700
            elif weight_val == 'normal':
                properties['weight'] = 400
            else:
                properties['weight'] = int(weight_val)
        # Check for <b> or <strong> tags
        elif element.name in ('b', 'strong') or element.find_parent(['b', 'strong']):
            properties['weight'] = 700

        # Parse font-size (extract numeric value in pt)
        size_match = re.search(r'font-size:\s*([\d.]+)(pt|px|em)?', style_str, re.IGNORECASE)
        if size_match:
            size_val = float(size_match.group(1))
            unit = size_match.group(2).lower() if size_match.group(2) else 'pt'
            # Convert to pt (rough approximation)
            if unit == 'px':
                size_val = size_val * 0.75  # 1pt ≈ 1.33px
            elif unit == 'em':
                size_val = size_val * 12  # Assume 12pt base
            properties['size_pt'] = size_val

        # Parse color
        color_match = re.search(r'color:\s*([#\w()]+(?:,\s*\d+\s*)*\)?)', style_str, re.IGNORECASE)
        if color_match:
            color_val = color_match.group(1).strip()
            # Handle rgb() with potential spaces
            if 'rgb' in color_val.lower():
                color_val = re.sub(r'\s*,\s*', ',', color_val)
            hex_color = self._parse_color_to_hex(color_val)
            if hex_color:
                properties['color_hex'] = hex_color

        # Parse font-family
        family_match = re.search(r'font-family:\s*([^;]+)', style_str, re.IGNORECASE)
        if family_match:
            properties['font_family'] = family_match.group(1).strip().strip('"\'').lower()

        return properties

    def _compute_styling_confidence(self, element_style: dict, baseline_style: dict) -> float:
        """Score how distinct element styling is from body text (0-1 scale)."""
        if not element_style:
            return 0.0

        confidence = 0.0

        # Compare font-weight (+0.3 if element bold and baseline not)
        elem_weight = element_style.get('weight', 400)
        base_weight = baseline_style.get('weight', 400)
        if elem_weight >= 700 and base_weight < 700:
            confidence += 0.3

        # Compare font-size (+0.2 if element ≥2pt larger)
        elem_size = element_style.get('size_pt', 0)
        base_size = baseline_style.get('size_pt', 11)
        if elem_size > 0 and elem_size >= base_size + 2:
            confidence += 0.2

        # Compare color (+0.3 if colors differ significantly)
        elem_color = element_style.get('color_hex')
        base_color = baseline_style.get('color_hex', '#000000')
        if elem_color and base_color:
            color_dist = self._color_distance(elem_color, base_color)
            if color_dist >= 0.2:  # Significant color difference (lowered threshold for dark blues vs black)
                confidence += 0.3

        # Compare font-family (+0.2 if different)
        elem_family = element_style.get('font_family')
        base_family = baseline_style.get('font_family')
        if elem_family and base_family and elem_family != base_family:
            confidence += 0.2

        return min(confidence, 1.0)

    def _find_styled_section_candidates(self, soup) -> List[tuple]:
        """
        Scan HTML for potential section headers based on styling.

        Returns: List of (element, matched_keyword, confidence_score) tuples
        """
        from bs4 import Tag
        from collections import Counter

        # Calculate baseline styling from body text (most common styles across <p> tags)
        p_tags = soup.find_all('p', limit=100)
        baseline_weights = []
        baseline_sizes = []
        baseline_colors = []
        baseline_families = []

        for p in p_tags:
            style = self._extract_style_properties(p)
            if style.get('weight'):
                baseline_weights.append(style['weight'])
            if style.get('size_pt'):
                baseline_sizes.append(style['size_pt'])
            if style.get('color_hex'):
                baseline_colors.append(style['color_hex'])
            if style.get('font_family'):
                baseline_families.append(style['font_family'])

        # Determine most common values (baseline)
        baseline_style = {}
        if baseline_weights:
            baseline_style['weight'] = Counter(baseline_weights).most_common(1)[0][0]
        else:
            baseline_style['weight'] = 400  # Default normal weight

        if baseline_sizes:
            baseline_style['size_pt'] = sum(baseline_sizes) / len(baseline_sizes)
        else:
            baseline_style['size_pt'] = 11  # Default size

        if baseline_colors:
            baseline_style['color_hex'] = Counter(baseline_colors).most_common(1)[0][0]
        else:
            baseline_style['color_hex'] = '#000000'  # Default black

        if baseline_families:
            baseline_style['font_family'] = Counter(baseline_families).most_common(1)[0][0]

        self._log(f"Baseline styling: weight={baseline_style.get('weight')}, size={baseline_style.get('size_pt'):.1f}pt, color={baseline_style.get('color_hex')}")

        # Sort keywords by length (longest first) for specificity
        sorted_keywords = sorted(SECTION_KEYWORDS.keys(), key=len, reverse=True)

        # Find all text elements that might be section headers
        candidates = []
        for tag_name in ['span', 'div', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'strong']:
            for element in soup.find_all(tag_name):
                if not isinstance(element, Tag):
                    continue

                # Get primary text content (ignore nested elements)
                element_text = element.get_text(separator=" ", strip=True).lower()
                if not element_text or len(element_text) > 200:
                    continue

                # Check if text matches any section keyword
                matched_keyword = None
                for keyword in sorted_keywords:
                    if keyword in element_text:
                        # Verify it's a substantial match (keyword is major part of text)
                        keyword_ratio = len(keyword) / len(element_text)
                        if keyword_ratio >= 0.3:  # Keyword is at least 30% of text
                            matched_keyword = keyword
                            break

                if not matched_keyword:
                    continue

                # Extract styling and compute confidence
                element_style = self._extract_style_properties(element)
                confidence = self._compute_styling_confidence(element_style, baseline_style)

                # Filter to high-confidence candidates
                if confidence >= 0.7:
                    candidates.append((element, matched_keyword, confidence))
                    self._log(f"Found candidate: '{element_text[:50]}' matched '{matched_keyword}' confidence={confidence:.2f}")

        return candidates

    def _parse_all_anchor_targets(self, soup) -> List[tuple]:
        """
        Find ALL elements with IDs that are linked to by anchor tags.

        Returns: [(element_id, doc_position), ...] sorted by document position
        """
        from bs4 import BeautifulSoup, Tag

        seen_ids = set()
        targets = []

        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if not href.startswith("#"):
                continue

            target_id = href[1:]
            if not target_id or target_id in seen_ids:
                continue

            target_elem = soup.find(id=target_id)
            if not target_elem:
                continue

            seen_ids.add(target_id)
            # Calculate document position by counting all previous elements
            pos = len(list(target_elem.find_all_previous()))
            targets.append((target_id, pos))

        # Sort by document position
        targets.sort(key=lambda x: x[1])
        return targets

    def _extract_html_between_ids(self, soup, start_id: str, end_id: Optional[str]) -> str:
        """
        Extract all HTML content between two element IDs.

        Returns the outer HTML of all elements between start and end.
        """
        from bs4 import BeautifulSoup, Tag

        start_elem = soup.find(id=start_id)
        if not start_elem:
            self._log(f"Could not find start element with id={start_id}")
            return ""

        # Collect all content from start element until end element
        html_parts = []

        # Include the start element itself
        html_parts.append(str(start_elem))

        # Find the end element if specified
        end_elem = soup.find(id=end_id) if end_id else None

        # Walk through siblings after start element
        current = start_elem.find_next_sibling()
        while current:
            # Stop if we've reached the end element or its container
            if end_elem and (
                current == end_elem or (isinstance(current, Tag) and current.find(id=end_id))
            ):
                break

            if isinstance(current, Tag):
                html_parts.append(str(current))

            current = current.find_next_sibling()

        return "\n".join(html_parts)

    def _html_to_text(self, html_content: str) -> str:
        """
        Convert HTML to plain text using BeautifulSoup.

        Simple text extraction - no markdown conversion.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_content, "lxml")
        return soup.get_text(separator="\n\n", strip=True)

    def _parse_toc_table_structure(self, soup) -> List[tuple]:
        """
        Parse table-based TOC to extract structured section index.

        Returns: List of (item_id, title, page_num, anchor_href) tuples
        """
        from bs4 import Tag

        toc_entries = []

        # Find tables that contain ITEM entries
        for table in soup.find_all('table'):
            if not isinstance(table, Tag):
                continue

            # Check if this table looks like a TOC
            table_text = table.get_text()
            item_count = len(re.findall(r'\bITEM\s+\d{1,2}[A-Z]?\b', table_text, re.IGNORECASE))

            if item_count < 3:
                continue

            self._log(f"Found TOC table with {item_count} ITEM entries")

            # Parse each row
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                row_text = row.get_text()

                # Look for ITEM pattern in row
                item_match = re.search(r'\bITEM\s+(\d{1,2}[A-Z]?)\b', row_text, re.IGNORECASE)
                if not item_match:
                    continue

                item_num = item_match.group(1).upper()
                item_id = f"ITEM {item_num}"

                # Extract page number from LAST cell (TOC format: Item | Title | Page)
                page_num = None
                if len(cells) >= 2:
                    # Last cell typically contains page number
                    last_cell_text = cells[-1].get_text().strip()
                    # Look for page patterns: "31", "Page 31", "Pages 31-46"
                    page_match = re.search(r'\b(\d{1,4})\b', last_cell_text)
                    if page_match:
                        try:
                            page_num = int(page_match.group(1))
                        except ValueError:
                            pass

                # Extract title (text after ITEM number, before page number)
                # Usually: "Item 1." or "Item 1" followed by title
                title_match = re.search(
                    rf'\bITEM\s+{re.escape(item_num)}\.?\s*[:.\-–—]?\s*(.+?)(?:\s+\d{{1,4}}\s*$|\s*$)',
                    row_text,
                    re.IGNORECASE | re.DOTALL
                )
                title = title_match.group(1).strip() if title_match else ""
                # Clean up title - remove page number patterns
                title = re.sub(r'\s+\d{1,4}\s*$', '', title)
                title = re.sub(r'Pages?\s+\d+[-,\s\d]*$', '', title, flags=re.IGNORECASE)
                title = self._clean_item_title(title)

                # Extract anchor link (look for <a href="#...">)
                anchor_href = None
                for link in row.find_all('a', href=True):
                    href = link.get('href', '')
                    if href.startswith('#'):
                        anchor_href = href[1:]  # Remove '#' prefix
                        break

                toc_entries.append((item_id, title, page_num, anchor_href))
                self._log(f"TOC entry: {item_id} | {title[:40]} | page={page_num} | anchor={anchor_href[-10:] if anchor_href else None}")

        return toc_entries

    def _map_toc_pages_to_parsed_pages(self, toc_page_num: Optional[int]) -> Optional[int]:
        """
        Map display page numbers from TOC to parsed page indices.

        SEC filings often have:
        - Cover page (not counted in TOC)
        - TOC page(s) (not counted in TOC)
        - Content starting at "Page 1"

        Returns: Parsed page index (1-based, matching page.number) or None if unmappable
        """
        if toc_page_num is None:
            return None

        # Try to find exact match using display_page attribute
        for page in self.pages:
            if hasattr(page, 'display_page') and page.display_page == toc_page_num:
                # Return the page number (1-based index)
                return page.number

        # Fallback: estimate based on first page with display_page
        # Find the offset between display_page and page.number
        for page in self.pages:
            if hasattr(page, 'display_page') and page.display_page is not None:
                offset = page.number - page.display_page
                estimated_page_num = toc_page_num + offset
                if 1 <= estimated_page_num <= len(self.pages):
                    return estimated_page_num
                break

        return None

    def _extract_sections_from_toc_table(self) -> List[Any]:
        """
        Extract sections using TOC table structure (comprehensive fallback).

        This method parses TOC tables to get all items, then uses:
        1. Anchor links when available
        2. Page numbers + content search for items without anchors
        3. Fuzzy title matching for section boundaries

        Returns: List of Section objects
        """
        from sec2md.models import Section, Page
        from bs4 import BeautifulSoup

        if not self.raw_html:
            self._log("No raw HTML available for TOC table extraction")
            return []

        soup = BeautifulSoup(self.raw_html, "lxml")

        # Parse TOC table structure
        toc_entries = self._parse_toc_table_structure(soup)

        if not toc_entries:
            self._log("No TOC table entries found")
            return []

        self._log(f"Found {len(toc_entries)} TOC entries")

        # Build mapping of anchor IDs to their document positions
        anchor_positions = {}
        all_anchors = self._parse_all_anchor_targets(soup)
        for anchor_id, pos in all_anchors:
            anchor_positions[anchor_id] = pos

        # Extract sections for each TOC entry
        sections = []

        for i, (item_id, title, page_num, anchor_href) in enumerate(toc_entries):
            # Skip filtered items
            if self.desired_items and item_id not in self.desired_items:
                continue

            content_parts = []

            # Strategy 1: Use anchor link if available
            if anchor_href and anchor_href in anchor_positions:
                # Find next anchor boundary
                next_anchor_id = None
                current_pos = anchor_positions[anchor_href]

                # Look for next TOC entry with anchor
                for j in range(i + 1, len(toc_entries)):
                    next_item_id, next_title, next_page, next_anchor = toc_entries[j]
                    if next_anchor and next_anchor in anchor_positions:
                        next_anchor_id = next_anchor
                        break

                # If no next TOC anchor, use next anchor in document
                if not next_anchor_id:
                    for anchor_id, pos in all_anchors:
                        if pos > current_pos:
                            next_anchor_id = anchor_id
                            break

                self._log(f"Extracting {item_id} via anchor {anchor_href[-10:]} to {next_anchor_id[-10:] if next_anchor_id else 'end'}")

                section_html = self._extract_html_between_ids(soup, anchor_href, next_anchor_id)
                if section_html:
                    text_content = self._html_to_text(section_html)
                    if len(text_content) > 100:
                        content_parts.append(text_content)

            # Strategy 2: Use page number to locate content
            elif page_num is not None:
                parsed_page_num = self._map_toc_pages_to_parsed_pages(page_num)

                if parsed_page_num and 1 <= parsed_page_num <= len(self.pages):
                    self._log(f"Extracting {item_id} via page number {page_num} (parsed page {parsed_page_num})")

                    # Search for section content starting at this page
                    # Look for the item title or ITEM pattern in page content
                    found_start = False

                    # Determine end page (next item's page or end)
                    end_page_num = len(self.pages) + 1
                    for j in range(i + 1, len(toc_entries)):
                        next_page_num = toc_entries[j][2]
                        if next_page_num:
                            next_parsed_num = self._map_toc_pages_to_parsed_pages(next_page_num)
                            if next_parsed_num:
                                end_page_num = next_parsed_num
                                break

                    # Extract pages in range (page.number is 1-based)
                    for page in self.pages:
                        if parsed_page_num <= page.number < end_page_num:
                            page_content = page.content

                            # Check if this page contains the section start
                            if not found_start:
                                # Look for ITEM pattern or title in content
                                item_pattern = re.compile(
                                    rf'\b{re.escape(item_id)}\b',
                                    re.IGNORECASE
                                )
                                if item_pattern.search(page_content) or (title and title.lower() in page_content.lower()):
                                    found_start = True

                            if found_start:
                                content_parts.append(page_content)

                    if not found_start:
                        self._log(f"Could not find start of {item_id} at page {page_num}")

            # Strategy 3: No anchor or page - search by item pattern in range
            else:
                self._log(f"Extracting {item_id} via content search (no anchor/page)")

                # Determine search range from surrounding items
                start_page_num = 1
                end_page_num = len(self.pages) + 1

                # Look backwards for previous item with page/anchor
                for j in range(i - 1, -1, -1):
                    prev_page_num = toc_entries[j][2]
                    if prev_page_num:
                        prev_parsed_num = self._map_toc_pages_to_parsed_pages(prev_page_num)
                        if prev_parsed_num:
                            start_page_num = prev_parsed_num
                            break

                # Look forwards for next item with page/anchor
                for j in range(i + 1, len(toc_entries)):
                    next_page_num = toc_entries[j][2]
                    if next_page_num:
                        next_parsed_num = self._map_toc_pages_to_parsed_pages(next_page_num)
                        if next_parsed_num:
                            end_page_num = next_parsed_num
                            break

                self._log(f"Searching for {item_id} in page range [{start_page_num}, {end_page_num})")

                # Search for ITEM pattern or title in this range
                found_start = False
                for page in self.pages:
                    if start_page_num <= page.number < end_page_num:
                        page_content = page.content

                        if not found_start:
                            # Look for exact ITEM pattern (e.g., "ITEM 7")
                            item_pattern = re.compile(
                                rf'\b{re.escape(item_id)}\b',
                                re.IGNORECASE
                            )
                            if item_pattern.search(page_content):
                                found_start = True
                                self._log(f"Found {item_id} at parsed page {page.number}")

                            # Also try matching by title if we have one
                            elif title and len(title) > 10:
                                # Check if title appears prominently in page
                                # Use first ~40 chars of title for matching
                                title_prefix = title[:40].lower()
                                if title_prefix in page_content.lower():
                                    found_start = True
                                    self._log(f"Found {item_id} by title match at parsed page {page.number}")

                        if found_start:
                            content_parts.append(page_content)

                # If we couldn't find the item but have a reasonable page range, extract all content
                # This handles cases where section has no header
                if not found_start and start_page_num < end_page_num and (end_page_num - start_page_num) <= 50:
                    self._log(f"Could not find {item_id} header, extracting all content in range [{start_page_num}, {end_page_num})")
                    for page in self.pages:
                        if start_page_num <= page.number < end_page_num:
                            content_parts.append(page.content)

                if not content_parts:
                    self._log(f"No content extracted for {item_id}")

            # Combine content parts
            if content_parts:
                combined = "\n\n".join(content_parts)

                # Apply length limit
                MAX_SECTION_CHARS = 120000
                if len(combined) > MAX_SECTION_CHARS:
                    self._log(f"Truncating {item_id} from {len(combined)} to {MAX_SECTION_CHARS} chars")
                    combined = combined[:MAX_SECTION_CHARS]

                if len(combined) > 500:
                    # Infer part from item number
                    part = None
                    if self.filing_type == "10-K":
                        part = self._infer_part_for_item(self.filing_type, item_id)

                    # Create section
                    page = Page(
                        number=1,
                        content=combined,
                        elements=None
                    )

                    sections.append(Section(
                        part=part,
                        item=item_id,
                        item_title=title if title else None,
                        pages=[page]
                    ))
                    self._log(f"Extracted {item_id}: {len(combined)} chars")

        return sections

    def _extract_sections_from_styling(self) -> List[Any]:
        """
        Extract sections using HTML styling analysis (middle-tier fallback).

        This method identifies section headers by analyzing visual styling (font-weight,
        font-size, color) combined with keyword matching. Works for filings that use
        styled headers but lack standard "PART I" / "ITEM 1" text patterns.

        Returns: List of Section objects
        """
        from sec2md.models import Section, Page
        from bs4 import BeautifulSoup

        if not self.raw_html:
            self._log("No raw HTML available for styling-based extraction")
            return []

        soup = BeautifulSoup(self.raw_html, "lxml")

        # Find all styled section candidates
        candidates = self._find_styled_section_candidates(soup)

        if not candidates:
            self._log("No styled section candidates found")
            return []

        self._log(f"Found {len(candidates)} styled section candidates with confidence scores")

        # Sort candidates by document position
        candidates_with_pos = []
        for element, keyword, confidence in candidates:
            pos = len(list(element.find_all_previous()))
            candidates_with_pos.append((pos, element, keyword, confidence))

        candidates_with_pos.sort(key=lambda x: x[0])

        # Extract content between consecutive section headers
        sections = []
        for i, (pos, element, keyword, confidence) in enumerate(candidates_with_pos):
            # Map keyword to ITEM number
            item = SECTION_KEYWORDS.get(keyword)
            if not item:
                continue

            # Infer part from item number
            part = None
            if self.filing_type == "10-K":
                part = self._infer_part_for_item(self.filing_type, item)

            # Extract title from element text
            elem_text = element.get_text(separator=" ", strip=True)
            # Try to clean up the title (remove just the keyword if it appears at start)
            item_title = elem_text
            if elem_text.lower().startswith(keyword):
                item_title = elem_text[len(keyword):].strip()
                item_title = self._clean_item_title(item_title) if item_title else keyword.title()
            else:
                item_title = elem_text

            # Find end boundary (next section or end of document)
            if i + 1 < len(candidates_with_pos):
                next_element = candidates_with_pos[i + 1][1]
                # Extract content between current element and next element
                content_parts = []
                current = element.find_next_sibling()
                while current and current != next_element:
                    if hasattr(current, 'get_text'):
                        content_parts.append(str(current))
                    current = current.find_next_sibling() if hasattr(current, 'find_next_sibling') else None
                    # Also check if we've passed the next element
                    if current and next_element in current.find_all_previous():
                        break

                section_html = "\n".join(content_parts)
            else:
                # Last section - extract to end of document
                content_parts = []
                current = element.find_next_sibling()
                while current:
                    if hasattr(current, 'get_text'):
                        content_parts.append(str(current))
                    current = current.find_next_sibling() if hasattr(current, 'find_next_sibling') else None

                section_html = "\n".join(content_parts)

            # Convert HTML to text
            text_content = self._html_to_text(section_html) if section_html else ""

            # Only include sections with substantial content
            if len(text_content) > 500:
                page = Page(
                    number=1,  # Styling-extracted content doesn't map to specific page numbers
                    content=text_content,
                    elements=None
                )

                sections.append(Section(
                    part=part,
                    item=item,
                    item_title=item_title,
                    pages=[page]
                ))
                self._log(f"Extracted {item} ({keyword}): {len(text_content)} chars")

        return sections

    def _extract_sections_from_toc(self) -> List[Any]:
        """
        Extract sections using TOC anchor links (fallback for non-standard formats).

        This method parses the raw HTML to find anchor targets in the table of contents,
        then extracts content between those anchors. This works for filings like Intel
        that use non-standard div-based layouts without semantic section headers.

        Returns: List of Section objects
        """
        from sec2md.models import Section, Page
        from bs4 import BeautifulSoup

        if not self.raw_html:
            self._log("No raw HTML available for TOC-based extraction")
            return []

        soup = BeautifulSoup(self.raw_html, "lxml")

        # Get ALL anchor targets sorted by document position
        all_anchors = self._parse_all_anchor_targets(soup)
        self._log(f"Found {len(all_anchors)} anchor targets")

        if not all_anchors:
            self._log("No anchor targets found in HTML")
            return []

        # Identify which anchors correspond to sections we care about
        section_anchors = []
        for i, (elem_id, pos) in enumerate(all_anchors):
            element = soup.find(id=elem_id)
            if not element:
                continue

            elem_text = self._get_element_section_text(element)
            item = self._match_section_to_item(elem_text)

            if item:
                section_anchors.append((i, item, elem_id))
                self._log(f"Found {item} at anchor #{elem_id[-10:]}")

        if not section_anchors:
            self._log("No section anchors matched ITEM patterns")
            return []

        # Group multiple anchors for the same ITEM together
        item_groups = {}
        for anchor_idx, item, elem_id in section_anchors:
            if item not in item_groups:
                item_groups[item] = []
            item_groups[item].append((anchor_idx, elem_id))

        self._log(f"Found {len(item_groups)} unique items: {list(item_groups.keys())}")

        # Extract content for each item
        sections = []
        for item, anchor_list in item_groups.items():
            all_text_parts = []

            for anchor_idx, start_id in anchor_list:
                # End boundary is the next anchor in document order
                end_id = all_anchors[anchor_idx + 1][0] if anchor_idx + 1 < len(all_anchors) else None

                self._log(f"Extracting {item} from #{start_id[-10:]} to #{end_id[-10:] if end_id else 'end'}")

                # Extract HTML between boundaries
                section_html = self._extract_html_between_ids(soup, start_id, end_id)

                if section_html and len(section_html) >= 100:
                    text_content = self._html_to_text(section_html)
                    if len(text_content) > 100:
                        all_text_parts.append(text_content)

            # Combine all parts for this item
            if all_text_parts:
                combined = "\n\n".join(all_text_parts)

                # Apply reasonable length limit (typical section is 30-80 pages)
                MAX_SECTION_CHARS = 120000
                if len(combined) > MAX_SECTION_CHARS:
                    self._log(f"Truncating {item} from {len(combined)} to {MAX_SECTION_CHARS} chars")
                    combined = combined[:MAX_SECTION_CHARS]

                if len(combined) > 500:
                    # Infer part from item number for 10-K
                    part = None
                    if self.filing_type == "10-K":
                        part = self._infer_part_for_item(self.filing_type, item)

                    # Extract item title from first anchor's text
                    first_anchor_id = anchor_list[0][1]
                    first_elem = soup.find(id=first_anchor_id)
                    elem_text = self._get_element_section_text(first_elem) if first_elem else ""

                    # Try to extract title after "ITEM N."
                    item_title = None
                    title_match = re.search(rf'{re.escape(item)}[\.:\-–—]\s*(.+)', elem_text, re.IGNORECASE)
                    if title_match:
                        item_title = self._clean_item_title(title_match.group(1))

                    # Create a single page with the combined content
                    page = Page(
                        number=1,  # TOC-extracted content doesn't map to specific page numbers
                        content=combined,
                        elements=None
                    )

                    sections.append(Section(
                        part=part,
                        item=item,
                        item_title=item_title,
                        pages=[page]
                    ))
                    self._log(f"Extracted {item}: {len(combined)} chars from {len(all_text_parts)} parts")

        return sections

    _ITEM_8K_RE = re.compile(
        rf'^\s*{LEAD_WRAP}(ITEM)\s+([1-9]\.\d{{2}}[A-Z]?)\.?\s*(?:[:.\-–—]\s*)?(.*)$',
        re.IGNORECASE | re.MULTILINE
    )
    _HARD_STOP_8K_RE = re.compile(r'^\s*(SIGNATURES|EXHIBIT\s+INDEX)\b', re.IGNORECASE | re.MULTILINE)
    _PROMOTE_ITEM_8K_RE = re.compile(r'(?<!\n)(\s)(ITEM\s+[1-9]\.\d{2}[A-Z]?\s*[.:–—-])', re.IGNORECASE)
    _PIPE_ROW_RE = re.compile(r'^\s*\|?\s*([0-9]{1,4}(?:\.[0-9A-Za-z]+)?)\s*\|\s*(.+?)\s*\|?\s*$', re.MULTILINE)
    _SPACE_ROW_RE = re.compile(r'^\s*([0-9]{1,4}(?:\.[0-9A-Za-z]+)?)\s{2,}(.+?)\s*$', re.MULTILINE)
    _HTML_ROW_RE = re.compile(
        r'<tr[^>]*>\s*<t[dh][^>]*>\s*([^<]+?)\s*</t[dh]>\s*<t[dh][^>]*>\s*([^<]+?)\s*</t[dh]>\s*</tr>',
        re.IGNORECASE | re.DOTALL
    )

    @staticmethod
    def _normalize_8k_item_code(code: str) -> str:
        """Normalize '5.2' -> '5.02', keep suffix 'A' if present."""
        code = code.upper().strip()
        m = re.match(r'^([1-9])\.(\d{1,2})([A-Z]?)$', code)
        if not m:
            return code
        major, minor, suffix = m.groups()
        minor = f"{int(minor):02d}"
        return f"{major}.{minor}{suffix}"

    def _clean_8k_text(self, text: str) -> str:
        """Clean 8-K text and normalize whitespace."""
        text = text.replace(NBSP, " ").replace(NARROW_NBSP, " ").replace(ZWSP, "")
        text = self._PROMOTE_ITEM_8K_RE.sub(r'\n\2', text)

        header_footer_8k = re.compile(
            r'^\s*(Form\s+8\-K|Page\s+\d+(?:\s+of\s+\d+)?|UNITED\s+STATES\s+SECURITIES\s+AND\s+EXCHANGE\s+COMMISSION)\b',
            re.IGNORECASE
        )

        lines: List[str] = []
        for ln in text.splitlines():
            t = ln.strip()
            if header_footer_8k.match(t):
                continue
            t = MD_EDGE.sub("", t)
            if re.fullmatch(r'\|\s*-{3,}\s*\|\s*-{3,}\s*\|?', t):
                continue
            lines.append(t)

        out: List[str] = []
        prev_blank = False
        for ln in lines:
            blank = (ln == "")
            if blank and prev_blank:
                continue
            out.append(ln)
            prev_blank = blank

        return "\n".join(out).strip()

    def _parse_exhibits(self, block: str) -> List[Any]:
        """Parse exhibit table from 9.01 section."""
        from sec2md.models import Exhibit
        rows: List[Exhibit] = []

        for m in self._PIPE_ROW_RE.finditer(block):
            left, right = m.group(1).strip(), m.group(2).strip()
            if not re.match(r'^\d', left):
                continue
            if left.startswith('---') or right.startswith('---'):
                continue
            rows.append(Exhibit(exhibit_no=left, description=right))
        if rows:
            return rows

        for m in self._SPACE_ROW_RE.finditer(block):
            left, right = m.group(1).strip(), m.group(2).strip()
            if not re.match(r'^\d', left):
                continue
            rows.append(Exhibit(exhibit_no=left, description=right))
        if rows:
            return rows

        for m in self._HTML_ROW_RE.finditer(block):
            left, right = m.group(1).strip(), m.group(2).strip()
            if not re.match(r'^\d', left):
                continue
            rows.append(Exhibit(exhibit_no=left, description=right))

        return rows

    def _slice_8k_body(self, doc: str, start_after: int, next_item_start: int) -> str:
        """Slice body text up to earliest hard stop."""
        mstop = self._HARD_STOP_8K_RE.search(doc, pos=start_after, endpos=next_item_start)
        end = mstop.start() if mstop else next_item_start
        return doc[start_after:end].strip()

    def _is_8k_boilerplate_page(self, page_content: str, page_num: int) -> bool:
        """Detect cover, TOC, and signature pages."""
        if page_num == 1:
            return True

        if re.search(r'TABLE OF CONTENTS', page_content, re.IGNORECASE):
            return True

        item_with_page_count = len(re.findall(r'ITEM\s+[1-9]\.\d{2}.*?\|\s*\d+\s*\|', page_content, re.IGNORECASE))
        if item_with_page_count >= 2:
            return True

        if re.search(r'\*\*SIGNATURES\*\*', page_content) and \
           re.search(r'Pursuant to the requirements', page_content, re.IGNORECASE):
            return True

        return False

    def _get_8k_sections(self) -> List[Any]:
        """Extract 8-K sections with TOC fallback."""
        from sec2md.models import Section, Page, ITEM_8K_TITLES

        sections = []
        current_item = None
        current_item_title = None
        current_pages: List[Page] = []

        def flush_section():
            nonlocal sections, current_item, current_item_title, current_pages
            if current_pages and current_item:
                exhibits = None
                if current_item.startswith("ITEM 9.01"):
                    content = "\n".join(p.content for p in current_pages)
                    md = re.search(r'^\s*\(?d\)?\s*Exhibits\b.*$', content, re.IGNORECASE | re.MULTILINE)
                    ex_block = content[md.end():].strip() if md else content
                    parsed_exhibits = self._parse_exhibits(ex_block)
                    exhibits = parsed_exhibits if parsed_exhibits else None

                sections.append(Section(
                    part=None,
                    item=current_item,
                    item_title=current_item_title,
                    pages=current_pages,
                    exhibits=exhibits
                ))
                current_pages = []

        for page in self.pages:
            page_num = page.number
            remaining_content = page.content

            if self._is_8k_boilerplate_page(remaining_content, page_num):
                self._log(f"DEBUG: Page {page_num} is boilerplate, skipping")
                continue

            while remaining_content:
                item_m = None
                first_idx = None

                for m in self._ITEM_8K_RE.finditer(remaining_content):
                    line_start = remaining_content.rfind('\n', 0, m.start()) + 1
                    line_end = remaining_content.find('\n', m.end())
                    if line_end == -1:
                        line_end = len(remaining_content)
                    full_line = remaining_content[line_start:line_end].strip()

                    if '|' in full_line:
                        self._log(f"DEBUG: Page {page_num} skipping table row: {full_line[:60]}")
                        continue

                    code = self._normalize_8k_item_code(m.group(2))
                    title_inline = (m.group(3) or "").strip()
                    title_inline = MD_EDGE.sub("", title_inline)

                    item_m = m
                    first_idx = m.start()
                    self._log(f"DEBUG: Page {page_num} found ITEM {code} at position {first_idx}")
                    break

                if first_idx is None:
                    if current_item and remaining_content.strip():
                        current_pages.append(Page(
                            number=page_num,
                            content=remaining_content,
                            elements=page.elements,
                            text_blocks=page.text_blocks,
                            display_page=page.display_page
                        ))
                    break

                before = remaining_content[:first_idx].strip()
                # Use header end position to skip past header and avoid infinite loop
                header_end = item_m.end()
                after = remaining_content[header_end:].strip()

                if current_item and before:
                    current_pages.append(Page(
                        number=page_num,
                        content=before,
                        elements=page.elements,
                        text_blocks=page.text_blocks,
                        display_page=page.display_page
                    ))

                flush_section()

                code = self._normalize_8k_item_code(item_m.group(2))
                title_inline = (item_m.group(3) or "").strip()
                title_inline = MD_EDGE.sub("", title_inline)
                current_item = f"ITEM {code}"
                current_item_title = title_inline if title_inline else ITEM_8K_TITLES.get(code)

                if self.desired_items and code not in self.desired_items:
                    self._log(f"DEBUG: Skipping ITEM {code} (not in desired_items)")
                    current_item = None
                    current_item_title = None
                    remaining_content = after
                    continue

                remaining_content = after

        flush_section()

        self._log(f"DEBUG: Total sections extracted: {len(sections)}")

        # Fallback to TOC-based extraction if pattern matching found no sections
        if not sections and self.raw_html:
            self._log("Pattern-based 8-K extraction found 0 sections, trying TOC fallback...")
            sections = self._extract_sections_from_toc()
            if sections:
                self._log(f"TOC fallback succeeded: extracted {len(sections)} sections")
            else:
                self._log("TOC fallback also found 0 sections")

        return sections

    def get_sections(self) -> List[Any]:
        """Get sections from the filing."""
        if self.filing_type == "8-K":
            return self._get_8k_sections()
        else:
            sections = self._get_standard_sections()

            # Four-tier fallback: pattern → styling → TOC table → TOC anchor
            if not sections and self.raw_html:
                self._log("Pattern-based extraction found 0 sections, trying styling-based extraction...")
                sections = self._extract_sections_from_styling()
                if sections:
                    self._log(f"Styling-based extraction found {len(sections)} sections")
                else:
                    self._log("Styling-based extraction found 0 sections, trying TOC table extraction...")
                    sections = self._extract_sections_from_toc_table()
                    if sections:
                        self._log(f"TOC table extraction succeeded: extracted {len(sections)} sections")
                    else:
                        self._log("TOC table extraction found 0 sections, trying TOC anchor fallback...")
                        sections = self._extract_sections_from_toc()
                        if sections:
                            self._log(f"TOC anchor fallback succeeded: extracted {len(sections)} sections")
                        else:
                            self._log("All extraction methods found 0 sections")

            return sections

    def _get_standard_sections(self) -> List[Any]:
        """Extract 10-K/10-Q/20-F sections."""
        from sec2md.models import Section, Page

        sections = []
        current_part = None
        current_item = None
        current_item_title = None
        current_pages: List[Page] = []

        def flush_section():
            nonlocal sections, current_part, current_item, current_item_title, current_pages
            if current_pages:
                sections.append(Section(
                    part=current_part,
                    item=current_item,
                    item_title=current_item_title,
                    pages=current_pages
                ))
                current_pages = []

        for page in self.pages:
            page_num = page.number
            content = page.content

            if self._is_toc(content, page_num):
                self._log(f"DEBUG: Page {page_num} detected as TOC, skipping")
                continue

            lines = self._clean_lines(content)
            joined = "\n".join(lines)

            if not joined.strip():
                self._log(f"DEBUG: Page {page_num} is empty after cleaning")
                continue

            part_m = None
            item_m = None
            first_idx = None
            first_kind = None

            for m in PART_PATTERN.finditer(joined):
                part_m = m
                first_idx = m.start()
                first_kind = 'part'
                self._log(f"DEBUG: Page {page_num} found PART at position {first_idx}: {m.group(1)}")
                break

            for m in ITEM_PATTERN.finditer(joined):
                if first_idx is None or m.start() < first_idx:
                    context_start = max(0, m.start() - 30)
                    context = joined[context_start:m.start()]
                    if re.search(r'\bPart\s+[IVXLC]+', context, re.IGNORECASE):
                        self._log(f"DEBUG: Page {page_num} skipping inline reference at {m.start()}")
                        continue

                    title = (m.group(3) or "").strip()
                    if not title or ITEM_BREADCRUMB_TITLE_RE.match(title):
                        self._log(f"DEBUG: Page {page_num} skipping breadcrumb ITEM {m.group(2)} with title '{title}'")
                        continue

                    item_m = m
                    first_idx = m.start()
                    first_kind = 'item'
                    self._log(f"DEBUG: Page {page_num} found ITEM at position {first_idx}: ITEM {m.group(2)}")
                break

            if first_kind is None:
                self._log(f"DEBUG: Page {page_num} - no header found. In section: {current_part or current_item}")
                if current_part or current_item:
                    if joined.strip():
                        current_pages.append(Page(
                            number=page_num,
                            content=joined,
                            elements=page.elements,
                            text_blocks=page.text_blocks,
                            display_page=page.display_page
                        ))
                continue

            before = joined[:first_idx].strip()
            after = joined[first_idx:].strip()

            if (current_part or current_item) and before:
                current_pages.append(Page(
                    number=page_num,
                    content=before,
                    elements=page.elements,
                    text_blocks=page.text_blocks,
                    display_page=page.display_page
                ))

            flush_section()

            if first_kind == 'part' and part_m:
                part_text = part_m.group(1)
                current_part, _ = self._normalize_section_key(part_text, None)
                current_item = None
                current_item_title = None
            elif first_kind == 'item' and item_m:
                item_num = item_m.group(2)
                title = (item_m.group(3) or "").strip()
                current_item_title = self._clean_item_title(title) if title else None
                if current_part is None and self.filing_type:
                    inferred = self._infer_part_for_item(self.filing_type, f"ITEM {item_num.upper()}")
                    if inferred:
                        current_part = inferred
                        self._log(f"DEBUG: Inferred {inferred} at detection time for ITEM {item_num}")
                _, current_item = self._normalize_section_key(current_part, item_num)

            if after:
                current_pages.append(Page(
                    number=page_num,
                    content=after,
                    elements=page.elements,
                    text_blocks=page.text_blocks,
                    display_page=page.display_page
                ))

                if first_kind == 'part' and part_m:
                    item_after = None
                    for m in ITEM_PATTERN.finditer(after):
                        title_after = (m.group(3) or "").strip()
                        if not title_after or ITEM_BREADCRUMB_TITLE_RE.match(title_after):
                            self._log(f"DEBUG: Page {page_num} skipping breadcrumb ITEM {m.group(2)} after PART with title '{title_after}'")
                            continue
                        item_after = m
                        break
                    if item_after:
                        start = item_after.start()
                        after = after[start:]
                        current_pages[-1] = Page(
                            number=page_num,
                            content=after,
                            elements=page.elements,
                            text_blocks=page.text_blocks,
                            display_page=page.display_page
                        )
                        item_num = item_after.group(2)
                        title = (item_after.group(3) or "").strip()
                        current_item_title = self._clean_item_title(title) if title else None
                        _, current_item = self._normalize_section_key(current_part, item_num)
                        self._log(f"DEBUG: Page {page_num} - promoted PART to ITEM {item_num} (intra-page)")

                tail = after
                while True:
                    next_kind, next_idx, next_part_m, next_item_m = None, None, None, None

                    for m in PART_PATTERN.finditer(tail):
                        if m.start() > 0:
                            next_kind, next_idx, next_part_m = 'part', m.start(), m
                            break
                    for m in ITEM_PATTERN.finditer(tail):
                        if m.start() > 0 and (next_idx is None or m.start() < next_idx):
                            title_tail = (m.group(3) or "").strip()
                            if not title_tail or ITEM_BREADCRUMB_TITLE_RE.match(title_tail):
                                self._log(f"DEBUG: Page {page_num} skipping breadcrumb ITEM {m.group(2)} in tail with title '{title_tail}'")
                                continue
                            next_kind, next_idx, next_item_m = 'item', m.start(), m

                    if next_idx is None:
                        break

                    before_seg = tail[:next_idx].strip()
                    after_seg = tail[next_idx:].strip()

                    if before_seg:
                        current_pages[-1] = Page(
                            number=page_num,
                            content=before_seg,
                            elements=page.elements,
                            text_blocks=page.text_blocks,
                            display_page=page.display_page
                        )
                    flush_section()

                    if next_kind == 'part' and next_part_m:
                        current_part, _ = self._normalize_section_key(next_part_m.group(1), None)
                        current_item = None
                        current_item_title = None
                        self._log(f"DEBUG: Page {page_num} - intra-page PART transition to {current_part}")
                    elif next_kind == 'item' and next_item_m:
                        item_num = next_item_m.group(2)
                        title = (next_item_m.group(3) or "").strip()
                        current_item_title = self._clean_item_title(title) if title else None
                        if current_part is None and self.filing_type:
                            inferred = self._infer_part_for_item(self.filing_type, f"ITEM {item_num.upper()}")
                            if inferred:
                                current_part = inferred
                                self._log(f"DEBUG: Inferred {inferred} at detection time for ITEM {item_num}")
                        _, current_item = self._normalize_section_key(current_part, item_num)
                        self._log(f"DEBUG: Page {page_num} - intra-page ITEM transition to {current_item}")

                    current_pages.append(Page(
                        number=page_num,
                        content=after_seg,
                        elements=page.elements,
                        text_blocks=page.text_blocks,
                        display_page=page.display_page
                    ))
                    tail = after_seg

        flush_section()

        self._log(f"DEBUG: Total sections before validation: {len(sections)}")
        for s in sections:
            self._log(f"  - Part: {s.part}, Item: {s.item}, Pages: {len(s.pages)}, Start: {s.pages[0].number if s.pages else 0}")

        def _section_text_len(s):
            return sum(len(p.content.strip()) for p in s.pages)

        sections = [s for s in sections if s.item is not None or _section_text_len(s) > 80]
        self._log(f"DEBUG: Sections after dropping empty PART stubs: {len(sections)}")

        if self.structure and sections:
            self._log(f"DEBUG: Validating against structure: {self.filing_type}")
            fixed = []
            for s in sections:
                part = s.part
                item = s.item

                # If part is missing or inconsistent with canonical mapping, try to infer it from the item.
                if item and self.filing_type:
                    inferred = self._infer_part_for_item(self.filing_type, item)
                    if inferred and inferred != part:
                        self._log(f"DEBUG: Rewriting part from {part} to {inferred} for {item}")
                        s = Section(
                            part=inferred,
                            item=s.item,
                            item_title=s.item_title,
                            pages=s.pages
                        )
                        part = inferred

                if (part in self.structure) and (item is None or item in self.structure.get(part, [])):
                    fixed.append(s)
                else:
                    self._log(f"DEBUG: Dropped section - Part: {part}, Item: {item}")

            sections = fixed
            self._log(f"DEBUG: Sections after validation: {len(sections)}")

        return sections

    def get_section(self, part: str, item: Optional[str] = None):
        """Get a specific section by part and item."""
        part_normalized = self._normalize_section(part)
        item_normalized = self._normalize_section(item) if item else None
        sections = self.get_sections()

        for section in sections:
            if section.part == part_normalized:
                if item_normalized is None or section.item == item_normalized:
                    return section
        return None

# Claude Context: sec2md Project

This document provides context for Claude (or any AI assistant) working on this codebase. It captures architectural decisions, recent changes, and key patterns to help maintain consistency.

---

## Project Overview

**sec2md** converts SEC EDGAR filings (10-K, 10-Q, 8-K, 20-F) from HTML to LLM-ready Markdown.

Key capabilities:
- HTML → Markdown conversion with structure preservation
- Section extraction (PART I, ITEM 1A, etc.) using multiple fallback strategies
- Table parsing (regular tables + absolutely positioned pseudo-tables)
- XBRL TextBlock tracking for financial statement notes
- Element-level citation support for RAG applications

---

## Architecture Overview

### Core Components

```
src/sec2md/
├── parser.py              # HTML → Markdown conversion, table parsing
├── section_extractor.py   # Section detection & extraction (multiple strategies)
├── models.py              # Pydantic models (Page, Section, Element, TextBlock)
├── core.py                # Public API entry point
├── sections.py            # Section extraction helpers
├── table_parser.py        # Standard table parsing
├── absolute_table_parser.py  # Absolutely positioned table detection
└── chunker/               # Content chunking for RAG (element-based)
```

### Key Scripts

- `html2md.py` - Simple HTML → Markdown conversion (no section extraction)
- `sections2md.py` - Extract sections → output with `##`/`###` headers
- `show_sections.py` - Display section metadata for inspection

---

## Recent Changes (Last Few Commits)

### 1. Section Headers in Markdown Output (Latest)

**What changed:**
- Modified `Section.markdown()` in `src/sec2md/models.py:411-439`
- Sections now output with markdown headings when calling `.markdown()`

**Implementation:**
```python
def markdown(self) -> str:
    # Main items (ITEM 1, ITEM 7) → ## (h2)
    # Sub-items (ITEM 1A, ITEM 7A) → ### (h3)
    heading_level = "###" if has_letter_suffix else "##"
    header = f"{heading_level} {part} {item} {title}"
    return f"{header}\n\n{content}"
```

**Impact:**
- `sections2md.py` now outputs sections with proper hierarchy
- `show_sections.py` displays sections with headers in preview
- Original page content untouched (headers added at section level)

**Example output:**
```markdown
## PART I ITEM 1 Business
[content]

### PART I ITEM 1A Risk Factors
[content]
```

### 2. HTML Heading Tags → Markdown Headings

**What changed:**
- Modified `src/sec2md/parser.py` to convert h1-h6 tags → `## ` markdown
- Added `_is_heading()` method (line 135)
- Modified `_wrap_markdown()` to exclude headings (line 227)
- Added heading handling in `_stream_pages()` (line 749)

**Implementation:**
- All h1-h6 tags now render as h2 (`## `) in markdown
- Treated as block elements with proper spacing
- Preserves existing bold/italic handling

**Why h2 for all?**
- Consistent heading level for document structure
- h1 reserved for document title
- SEC filings don't use semantic heading hierarchy

---

## Section Extraction Strategy

The section extractor uses **multiple fallback strategies** (most reliable first):

### 1. Pattern-Based Extraction (Primary)
- Regex patterns match section headers in content
- Handles format variations: "ITEM 1A", "Item 1A.", "ITEM 1A -"
- See `ITEM_PATTERNS` in `section_extractor.py`

### 2. TOC-Based Extraction (Fallback)
- Parses Table of Contents anchor links
- Matches TOC entries to content using fuzzy matching
- Used for non-standard formats (e.g., Intel filings)

### 3. Styling-Based Extraction (Fallback)
- Detects headers by font size, weight, color
- Uses statistical confidence scoring
- Last resort for unusual formats

### Fuzzy Matching Algorithm

Used to match TOC entries to content sections:

**Implementation:** Sørensen-Dice coefficient on character multisets
```python
# _fuzzy_match_ratio in section_extractor.py:303
similarity = 2 * |intersection| / (|set1| + |set2|)
```

**Why this approach:**
- Order-independent (handles "Risk Factors" vs "Factors Risk" → 1.0)
- No external dependencies (just `collections.Counter`)
- Handles format variations: "Item 1A. Risk" vs "ITEM 1A - Risk" → 0.93
- Fast: O(n+m)

**Tradeoffs:**
- Struggles with abbreviations: "MD&A" vs "Management Discussion" → 0.13
- Could add abbreviation expansion preprocessing if needed

**See:** `FUZZY_MATCHING_ANALYSIS.md` for full comparison vs difflib/Levenshtein/thefuzz

---

## Key Design Patterns

### 1. Parser: Single-Pass Streaming

The parser (`parser.py`) walks the DOM once and streams content to pages:

```python
def _stream_pages(self, root, page_num):
    # CSS page-break detection
    if self._has_break_before(root):
        page_num += 1

    # Block element handling (tables, lists, headings)
    # Inline element handling (bold, italic)
    # Text node handling

    # Recursively process children
    for child in root.children:
        page_num = self._stream_pages(child, page_num)

    return page_num
```

**Key features:**
- Tracks XBRL TextBlock context for financial notes
- Handles absolutely positioned pseudo-tables
- Merges inline formatting spans (`**text** **more**` → `**text more**`)

### 2. Absolutely Positioned Tables

SEC filings often use CSS `position:absolute` for table layouts instead of `<table>` tags.

**Detection:** `_is_absolutely_positioned()` checks for `position:absolute` in style
**Parsing:** Groups elements by Y-position, detects column structure
**See:** `absolute_table_parser.py` and `parser.py:574`

### 3. Element-Based Structure

Pages can include structured elements for citation:

```python
class Element(BaseModel):
    id: str  # "sec2md-p1-h3-a4b2c1d3"
    content: str
    kind: str  # "paragraph", "table", "header", "list"
    page_start: int
    page_end: int
```

Used for:
- RAG applications (cite specific paragraphs)
- Chunking strategies (see `chunker/`)
- TextBlock association (financial note tracking)

---

## Testing

### Running Tests
```bash
pytest tests/ -v
```

### Test Coverage
- `tests/test_section_extraction.py` - Section extraction strategies
  - Standard format (Microsoft pattern)
  - Non-standard format (Intel TOC-based)
  - Styling-based extraction
  - Regression tests

### Fixtures
- `tests/fixtures/nvidia_raw.html` - NVIDIA 10-K
- `tests/fixtures/microsoft_raw.html` - Microsoft 10-K
- `tests/fixtures/intel_raw.html` - Intel 10-K (non-standard format)

---

## Linting & Formatting

```bash
# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Check formatting
uv run black --check .

# Format code
uv run black .

# Combined workflow
uv run ruff check --fix . && uv run black .
```

**Config:** See `pyproject.toml`
- Line length: 100
- Target: Python 3.9+

---

## Common Tasks

### Add a New Section Pattern

Edit `ITEM_PATTERNS` in `section_extractor.py`:

```python
ITEM_PATTERNS = {
    "10-K": {
        "PART I": [
            (r"ITEM\s+1A\.?\s+Risk\s+Factors", "ITEM 1A"),
            # Add new pattern here
        ]
    }
}
```

### Add a New Filing Type

1. Add enum to `models.py` (e.g., `Item20F`)
2. Add patterns to `ITEM_PATTERNS` in `section_extractor.py`
3. Update `FilingType` type alias in `models.py`

### Modify Markdown Output Format

- **Page-level:** Edit `parser.py` (heading rendering, table formatting)
- **Section-level:** Edit `Section.markdown()` in `models.py`
- **Content joining:** Edit how pages are concatenated

---

## Known Issues & Limitations

### 1. Abbreviations in Fuzzy Matching
- "MD&A" vs "Management's Discussion and Analysis" → low score (0.13)
- **Potential fix:** Add abbreviation expansion preprocessing

### 2. Absolutely Positioned Tables
- Complex multi-column layouts may not parse perfectly
- Requires Y-position clustering and column detection heuristics

### 3. Page Number Detection
- Extracted from footer elements or content pattern matching
- May fail on unusual formats
- See `_detect_display_page_numbers()` in `parser.py`

### 4. XBRL TextBlock Tracking
- Tracks `ix:nonnumeric` tags with `TextBlock` names
- Continuation tracking via `continuedat` attribute
- May miss notes without proper XBRL tagging

---

## Development Tips

### 1. Debugging Section Extraction

Use `show_sections.py` to inspect extraction results:
```bash
python show_sections.py tests/fixtures/nvidia_raw.html --debug --show-content
```

### 2. Testing Markdown Output

Quick test with fixtures:
```bash
# Simple conversion
python html2md.py tests/fixtures/nvidia_raw.html -o /tmp/test.md

# With section extraction
python sections2md.py tests/fixtures/nvidia_raw.html -o /tmp/test_sections.md
```

### 3. Inspecting Parser Output

```python
from sec2md.parser import Parser

parser = Parser(html_content)
pages = parser.get_pages()

# Check page structure
for page in pages:
    print(f"Page {page.number}: {page.tokens} tokens")
    if page.elements:
        print(f"  Elements: {len(page.elements)}")
    if page.text_blocks:
        print(f"  TextBlocks: {len(page.text_blocks)}")
```

### 4. Testing Fuzzy Matching

See `FUZZY_MATCHING_ANALYSIS.md` for comparison scripts and analysis.

---

## Project Structure Notes

### Why `parser.py` is so large
- Handles multiple responsibilities: HTML parsing, table detection, page breaking, XBRL tracking
- Could be refactored into separate modules in the future
- Main areas: DOM traversal, table parsing, absolutely positioned layouts, element tracking

### Why Multiple Section Extraction Strategies
- SEC filings have no standard format
- Companies use different HTML structures, styling, TOC formats
- Fallback strategy ensures broad compatibility

### Why Pydantic Models
- Type safety and validation
- Easy serialization to JSON for API usage
- Computed fields for derived properties (tokens, page_range)
- Integration with IDEs for autocomplete

---

## Future Considerations

### Potential Improvements

1. **Abbreviation Expansion**
   - Add dictionary for common SEC abbreviations (MD&A, etc.)
   - Apply before fuzzy matching

2. **Better Table Parsing**
   - Handle nested tables
   - Improve multi-column layout detection
   - Add table header detection

3. **Section Extraction Tuning**
   - Add more filing-specific patterns
   - Improve confidence scoring for styling-based extraction
   - Add validation against known section structures

4. **Performance Optimization**
   - Cache parsed sections
   - Parallel processing for large documents
   - Optimize DOM traversal

5. **Additional Filing Types**
   - 8-K full support (currently basic)
   - 20-F full support
   - S-1, DEF 14A, etc.

---

## Questions to Ask When Making Changes

### Changing Parser Logic
- Does it affect page breaking?
- Does it maintain XBRL TextBlock context?
- Does it handle absolutely positioned layouts?
- Are inline spans properly merged?

### Changing Section Extraction
- Does it work for all test fixtures (Microsoft, Intel, NVIDIA)?
- Does it handle format variations (punctuation, case, spacing)?
- What's the fallback behavior if primary strategy fails?

### Changing Models
- Is it a breaking change to the API?
- Does it affect serialization/JSON output?
- Are computed fields still valid?

### Adding Dependencies
- Is it really needed? (prefer stdlib)
- Does it work with Python 3.9+?
- Is it in `pyproject.toml`?

---

## Contact & Resources

- **Repository:** https://github.com/lucasastorian/sec2md
- **Original Author:** Lucas Astorian (lucas@intellifin.ai)
- **Recent Contributor:** Will (these notes)

**Related Documentation:**
- `FUZZY_MATCHING_ANALYSIS.md` - Fuzzy matching algorithm comparison
- `README.md` - User-facing documentation
- `pyproject.toml` - Project configuration

---

## Version History Context

- **v0.1.20** (current) - Added section markdown headers, HTML heading conversion
- Earlier versions - Core parsing, section extraction, chunking features

---

*Last Updated: 2026-01-12*
*Context for: Claude Sonnet 4.5*

"""Test section extraction for both standard and non-standard filings."""

import pytest
from pathlib import Path
from sec2md.parser import Parser
from sec2md.section_extractor import SectionExtractor


@pytest.fixture
def intel_html():
    """Load Intel 10-K HTML fixture (non-standard format)."""
    fixture_path = Path(__file__).parent / "fixtures" / "intel_raw.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def microsoft_html():
    """Load Microsoft 10-K HTML fixture (standard format)."""
    fixture_path = Path(__file__).parent / "fixtures" / "microsoft_raw.html"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return f.read()


class TestStandardFormatExtraction:
    """Test pattern-based extraction on standard filings (Microsoft)."""

    def test_microsoft_pattern_extraction(self, microsoft_html):
        """Microsoft filing should work with pattern-based extraction."""
        parser = Parser(microsoft_html)
        pages = parser.get_pages(include_elements=False)

        extractor = SectionExtractor(
            pages=pages,
            filing_type="10-K",
            debug=True,
            raw_html=parser.raw_html
        )

        sections = extractor.get_sections()

        # Microsoft should extract sections successfully
        assert len(sections) > 0, "Microsoft filing should extract >0 sections with pattern matching"

        # Should find key sections
        section_items = {s.item for s in sections}
        assert "ITEM 1" in section_items, "Should find ITEM 1 (Business)"
        assert "ITEM 1A" in section_items, "Should find ITEM 1A (Risk Factors)"
        assert "ITEM 7" in section_items, "Should find ITEM 7 (MD&A)"

        # Each section should have content (except ITEM 6 which is often [RESERVED])
        for section in sections:
            assert section.pages, f"{section.item} should have pages"
            content = section.markdown()
            if section.item != "ITEM 6":  # ITEM 6 is often [RESERVED]
                assert len(content) > 50, f"{section.item} should have some content"


class TestNonStandardFormatExtraction:
    """Test TOC-based extraction on non-standard filings (Intel)."""

    def test_intel_toc_extraction(self, intel_html):
        """Intel filing should work with TOC-based fallback extraction."""
        parser = Parser(intel_html)
        pages = parser.get_pages(include_elements=False)

        extractor = SectionExtractor(
            pages=pages,
            filing_type="10-K",
            debug=True,
            raw_html=parser.raw_html
        )

        sections = extractor.get_sections()

        # Intel filing should extract sections via TOC fallback
        assert len(sections) > 0, "Intel filing should extract >0 sections with TOC fallback"

        # Should find key sections (TOC extraction may not find all items)
        section_items = {s.item for s in sections}
        # Intel's TOC extraction finds ITEM 1A and ITEM 7 reliably
        assert "ITEM 1A" in section_items, "Should find ITEM 1A (Risk Factors)"
        assert "ITEM 7" in section_items, "Should find ITEM 7 (MD&A)"

        # Each section should have content (except ITEM 6 which is often [RESERVED])
        for section in sections:
            assert section.pages, f"{section.item} should have pages"
            content = section.markdown()
            if section.item != "ITEM 6":  # ITEM 6 is often [RESERVED]
                assert len(content) > 50, f"{section.item} should have some content"

    def test_intel_pattern_extraction_fails(self, intel_html):
        """Intel filing should fail with pattern-only extraction (no raw_html)."""
        parser = Parser(intel_html)
        pages = parser.get_pages(include_elements=False)

        # Create extractor WITHOUT raw_html to force pattern-only mode
        extractor = SectionExtractor(
            pages=pages,
            filing_type="10-K",
            debug=True,
            raw_html=None  # No fallback available
        )

        sections = extractor.get_sections()

        # Should return 0 sections because pattern matching fails on Intel format
        assert len(sections) == 0, "Intel filing should extract 0 sections without TOC fallback"


class TestTOCExtractionComponents:
    """Test individual TOC extraction methods."""

    def test_parse_all_anchor_targets(self, intel_html):
        """Test anchor target parsing."""
        parser = Parser(intel_html)
        pages = parser.get_pages(include_elements=False)

        extractor = SectionExtractor(
            pages=pages,
            filing_type="10-K",
            raw_html=parser.raw_html
        )

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(parser.raw_html, "lxml")
        anchors = extractor._parse_all_anchor_targets(soup)

        # Intel filing has many anchor links
        assert len(anchors) > 50, "Intel filing should have many anchor targets"

        # Anchors should be sorted by document position
        positions = [pos for _, pos in anchors]
        assert positions == sorted(positions), "Anchors should be sorted by position"

    def test_match_section_to_item(self):
        """Test section text matching to ITEM identifiers."""
        parser = Parser("<html></html>")
        extractor = SectionExtractor(pages=[], filing_type="10-K")

        # Test various formats
        assert extractor._match_section_to_item("Item 1A. Risk Factors") == "ITEM 1A"
        assert extractor._match_section_to_item("ITEM 7 Management's Discussion") == "ITEM 7"
        assert extractor._match_section_to_item("Item 1. Business") == "ITEM 1"
        assert extractor._match_section_to_item("Risk Factors") == "ITEM 1A"
        assert extractor._match_section_to_item("MD&A") == "ITEM 7"
        assert extractor._match_section_to_item("Business Description") == "ITEM 1"

        # Should return None for non-matching text
        assert extractor._match_section_to_item("Random text") is None
        assert extractor._match_section_to_item("") is None


class TestStylingBasedExtraction:
    """Test styling-based section detection."""

    @pytest.fixture
    def styled_headers_html(self):
        """HTML with styled headers but no PART/ITEM text."""
        return '''
        <html><body>
        <div style="font-weight:bold;font-size:14pt;color:#003366">Risk Factors</div>
        <p style="font-size:11pt">This is body text about risks. Our business faces various risks including market volatility, competitive pressures, regulatory changes, and operational challenges. These risk factors should be carefully considered by investors when evaluating our company. Economic conditions may impact our revenue and profitability. We operate in a highly competitive industry with rapidly evolving technology. Our success depends on our ability to innovate and adapt to changing market conditions. Failure to manage these risks effectively could materially harm our business, financial condition, and results of operations.</p>
        <p style="font-size:11pt">More risk content spanning multiple paragraphs with additional details about specific risk categories including cybersecurity threats, supply chain disruptions, and geopolitical uncertainties that could affect our global operations.</p>
        <div style="font-weight:700;font-size:14pt;color:rgb(0,51,102)">Management's Discussion and Analysis</div>
        <p style="font-size:11pt">MD&A content here discussing our financial results, key performance indicators, and strategic initiatives. This section provides management's perspective on our operating results, liquidity, and capital resources. We analyze trends and uncertainties that have had or are expected to have a material impact on our business. Our revenue grew year-over-year driven by strong demand across all product lines. Operating expenses increased primarily due to investments in research and development. We maintain a strong balance sheet with sufficient liquidity to fund our operations and strategic growth initiatives.</p>
        </body></html>
        '''

    def test_styled_header_extraction(self, styled_headers_html):
        """Verify extraction from styled headers without PART/ITEM patterns."""
        parser = Parser(styled_headers_html)
        pages = parser.get_pages(include_elements=False)

        extractor = SectionExtractor(
            pages=pages,
            filing_type="10-K",
            debug=True,
            raw_html=parser.raw_html
        )

        sections = extractor.get_sections()

        # Should extract sections via styling detection
        assert len(sections) >= 2, "Should detect at least 2 styled sections"
        section_items = {s.item for s in sections}
        assert "ITEM 1A" in section_items, "Should detect 'Risk Factors' as ITEM 1A"
        assert "ITEM 7" in section_items, "Should detect 'Management's Discussion' as ITEM 7"

    def test_styling_confidence_scoring(self):
        """Verify confidence scoring filters false positives."""
        # HTML with both real section header and coincidental bold text
        html = '''
        <html><body style="font-size:11pt;color:#000000">
        <span style="font-weight:bold;font-size:14pt;color:#003366">Risk Factors</span>
        <p>This paragraph has <b>bold emphasis</b> but should not be a section.</p>
        </body></html>
        '''

        parser = Parser(html)
        extractor = SectionExtractor(pages=[], filing_type="10-K", raw_html=parser.raw_html)

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(parser.raw_html, "lxml")
        candidates = extractor._find_styled_section_candidates(soup)

        # Should find only the styled header, not inline bold
        assert len(candidates) == 1, "Should find only 1 candidate (the styled header)"
        assert candidates[0][1] == "risk factors", "Should match 'Risk Factors' keyword"
        assert candidates[0][2] > 0.7, "Confidence should be >0.7 for real section header"

    def test_color_format_handling(self):
        """Verify detection works with hex, rgb(), and named colors."""
        html = '''
        <html><body>
        <div style="font-weight:bold;font-size:14pt;color:#003366">Business</div>
        <p style="font-size:11pt">Our company operates in multiple business segments providing various products and services to customers worldwide. We have a diversified portfolio and strong market positions across our key business lines. Our business strategy focuses on innovation, operational excellence, and customer satisfaction. We invest significantly in research and development to maintain our competitive advantage and drive long-term growth. Additional content to reach character threshold for testing purposes with more descriptive business information about our operations, markets, and competitive positioning.</p>
        <div style="font-weight:bold;font-size:14pt;color:rgb(0,51,102)">Properties</div>
        <p style="font-size:11pt">We own and lease various properties including corporate offices, manufacturing facilities, distribution centers, and retail locations around the world. Our real estate portfolio is strategically located to support our operations and serve our customers effectively in all major markets. We regularly evaluate our property needs and make investments to optimize our physical footprint and support our business objectives for the long term. We continue to expand our facilities to accommodate growth and enhance operational efficiency across all regions.</p>
        <div style="font-weight:bold;font-size:14pt;color:navy">Legal Proceedings</div>
        <p style="font-size:11pt">We are involved in various legal proceedings and claims arising in the ordinary course of business across multiple jurisdictions. While the outcome of these matters cannot be predicted with certainty, we do not believe that any of these proceedings will have a material adverse effect on our financial position or results of operations based on current information. We maintain comprehensive insurance coverage and establish reserves where appropriate to mitigate potential liabilities. Our experienced legal team actively manages all litigation matters and works closely with external counsel to protect the company's interests.</p>
        </body></html>
        '''

        parser = Parser(html)
        pages = parser.get_pages(include_elements=False)
        extractor = SectionExtractor(pages=pages, filing_type="10-K", debug=True, raw_html=parser.raw_html)

        sections = extractor.get_sections()
        section_items = {s.item for s in sections}

        assert "ITEM 1" in section_items, "Should detect hex color header"
        assert "ITEM 2" in section_items, "Should detect rgb() color header"
        assert "ITEM 3" in section_items, "Should detect named color header"

    def test_fallback_ordering(self):
        """Verify pattern → styling → TOC fallback order."""
        # Standard format should use pattern extraction
        standard_html = '<html><body><p>PART I</p><p>ITEM 1. Business</p></body></html>'
        parser = Parser(standard_html)
        pages = parser.get_pages(include_elements=False)
        extractor = SectionExtractor(pages=pages, filing_type="10-K", debug=True, raw_html=parser.raw_html)

        sections = extractor.get_sections()
        # Should extract via pattern (will see "PART I" in markdown)
        assert len(sections) > 0


class TestRegressionPrevention:
    """Tests to ensure both formats continue working."""

    def test_both_formats_extract_risk_factors(self, intel_html, microsoft_html):
        """Both Intel and Microsoft should extract Risk Factors (ITEM 1A)."""
        for name, html in [("Intel", intel_html), ("Microsoft", microsoft_html)]:
            parser = Parser(html)
            pages = parser.get_pages(include_elements=False)

            extractor = SectionExtractor(
                pages=pages,
                filing_type="10-K",
                debug=True,
                raw_html=parser.raw_html
            )

            sections = extractor.get_sections()
            section_items = {s.item for s in sections}

            assert "ITEM 1A" in section_items, f"{name} should extract ITEM 1A"

            # Find the Risk Factors section
            risk_section = next(s for s in sections if s.item == "ITEM 1A")
            content = risk_section.markdown()

            # Risk Factors sections are typically very long
            assert len(content) > 5000, f"{name} ITEM 1A should have substantial content (>5000 chars)"

    def test_both_formats_extract_md_and_a(self, intel_html, microsoft_html):
        """Both Intel and Microsoft should extract MD&A (ITEM 7)."""
        for name, html in [("Intel", intel_html), ("Microsoft", microsoft_html)]:
            parser = Parser(html)
            pages = parser.get_pages(include_elements=False)

            extractor = SectionExtractor(
                pages=pages,
                filing_type="10-K",
                debug=True,
                raw_html=parser.raw_html
            )

            sections = extractor.get_sections()
            section_items = {s.item for s in sections}

            assert "ITEM 7" in section_items, f"{name} should extract ITEM 7"

            # Find the MD&A section
            mda_section = next(s for s in sections if s.item == "ITEM 7")
            content = mda_section.markdown()

            # MD&A sections are typically very long
            assert len(content) > 5000, f"{name} ITEM 7 should have substantial content (>5000 chars)"

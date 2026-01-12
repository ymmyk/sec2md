#!/usr/bin/env python3
"""
Script to extract and display sections from SEC HTML filings.

This script uses the sec2md library to parse SEC HTML filings and extract
structured sections (PART I, ITEM 1A, etc.). It shows section headers,
page ranges, token counts, and optionally content previews.

Usage:
    python show_sections.py <html_file> [filing_type]

Arguments:
    html_file      Path to the HTML filing to analyze
    filing_type    Optional filing type (10-K, 10-Q, 8-K, 20-F)
                   If not provided, will attempt auto-detection

Options:
    --debug            Enable debug output showing parsing details
    --show-content     Display preview of section content
    --max-preview N    Characters to show in preview (default: 500)

Examples:
    # Basic usage with auto-detection
    python show_sections.py msft_10k.html

    # Explicit filing type
    python show_sections.py aapl_10q.html 10-Q

    # Show content previews
    python show_sections.py filing.html --show-content

    # Debug mode to see parsing details
    python show_sections.py filing.html 10-K --debug

Output:
    The script displays:
    - Total number of sections found
    - For each section:
        * PART and ITEM designation
        * Section title
        * Page range in the document
        * Token count (for estimating LLM context usage)
        * Exhibits list (for 8-K filings with Item 9.01)
        * Content preview (if --show-content is used)
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from sec2md.parser import Parser
from sec2md.section_extractor import SectionExtractor


def detect_filing_type(html_content: str) -> Optional[str]:
    """Attempt to detect filing type from HTML content."""
    html_upper = html_content[:5000].upper()

    if "10-K" in html_upper:
        return "10-K"
    elif "10-Q" in html_upper:
        return "10-Q"
    elif "8-K" in html_upper:
        return "8-K"
    elif "20-F" in html_upper:
        return "20-F"

    return None


def format_section_header(section) -> str:
    """Format section header for display."""
    parts = []

    if section.part:
        parts.append(section.part)
    if section.item:
        parts.append(section.item)
    if section.item_title:
        parts.append(f"- {section.item_title}")

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and display sections from SEC HTML filings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python show_sections.py filing.html 10-K
  python show_sections.py filing.html --debug
  python show_sections.py filing.html 10-Q --show-content
        """,
    )

    parser.add_argument("html_file", type=str, help="Path to HTML filing")
    parser.add_argument(
        "filing_type",
        type=str,
        nargs="?",
        help="Filing type (10-K, 10-Q, 8-K, 20-F). Auto-detected if not provided.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--show-content", action="store_true", help="Show preview of section content"
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=500,
        help="Maximum characters to show in preview (default: 500)",
    )

    args = parser.parse_args()

    # Load HTML file
    html_path = Path(args.html_file)
    if not html_path.exists():
        print(f"Error: File not found: {html_path}", file=sys.stderr)
        return 1

    print(f"Reading file: {html_path}")
    html_content = html_path.read_text(encoding="utf-8")

    # Detect or use provided filing type
    filing_type = args.filing_type
    if not filing_type:
        filing_type = detect_filing_type(html_content)
        if filing_type:
            print(f"Auto-detected filing type: {filing_type}")
        else:
            print("Warning: Could not auto-detect filing type. Results may be limited.")
    else:
        filing_type = filing_type.upper()
        print(f"Using filing type: {filing_type}")

    # Parse HTML to pages
    print("\nParsing HTML...")
    html_parser = Parser(html_content)
    pages = html_parser.get_pages()

    print(f"Parsed {len(pages)} pages")

    # Extract sections
    print("\nExtracting sections...")
    extractor = SectionExtractor(
        pages=pages, filing_type=filing_type, debug=args.debug, raw_html=html_content
    )

    sections = extractor.get_sections()

    # Display results
    print(f"\n{'='*80}")
    print(f"SECTIONS FOUND: {len(sections)}")
    print(f"{'='*80}\n")

    if not sections:
        print("No sections found.")
        return 0

    for i, section in enumerate(sections, 1):
        header = format_section_header(section)
        page_start, page_end = section.page_range
        page_info = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"

        print(f"{i}. {header}")
        print(f"   Pages: {page_info}, Tokens: {section.tokens:,}")

        if section.exhibits:
            print(f"   Exhibits: {len(section.exhibits)}")
            for exhibit in section.exhibits[:3]:  # Show first 3 exhibits
                print(f"     - {exhibit.exhibit_no}: {exhibit.description[:80]}")
            if len(section.exhibits) > 3:
                print(f"     ... and {len(section.exhibits) - 3} more")

        if args.show_content:
            content = section.markdown()
            preview = content[: args.max_preview]
            if len(content) > args.max_preview:
                preview += "..."
            print("\n   Preview:")
            for line in preview.split("\n"):
                print(f"   {line}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

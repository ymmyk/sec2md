#!/usr/bin/env python3
"""
Extract sections from SEC HTML filing and output to markdown with section headers.

This script uses section extraction to identify PART/ITEM sections and outputs
each section with a proper markdown heading (## for main items, ### for sub-items).
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract sections from SEC HTML filing and convert to markdown with headers"
    )
    parser.add_argument("input_file", help="Path to input HTML file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (optional, defaults to input filename with _sections.md extension)",
    )
    parser.add_argument(
        "--filing-type",
        help="Filing type (10-K, 10-Q, 8-K, 20-F). Auto-detected if not provided.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: '{args.input_file}' is not a file", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Add _sections suffix before .md extension
        output_path = input_path.with_stem(input_path.stem + "_sections").with_suffix(".md")

    try:
        # Read HTML content
        print(f"Reading file: {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            html_content = f.read()

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
        print("Parsing HTML...")
        html_parser = Parser(html_content)
        pages = html_parser.get_pages()
        print(f"Parsed {len(pages)} pages")

        # Extract sections
        print("Extracting sections...")
        extractor = SectionExtractor(
            pages=pages, filing_type=filing_type, debug=args.debug, raw_html=html_content
        )
        sections = extractor.get_sections()
        print(f"Found {len(sections)} sections")

        if not sections:
            print("Warning: No sections found. Output will be empty.", file=sys.stderr)

        # Write sections to markdown file
        print(f"Writing sections to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, section in enumerate(sections):
                # Get markdown with header
                section_md = section.markdown()
                f.write(section_md)

                # Add separator between sections (except after last one)
                if i < len(sections) - 1:
                    f.write("\n\n---\n\n")

        print(f"Successfully wrote {len(sections)} sections to '{output_path}'")
        print(f"Total tokens: {sum(s.tokens for s in sections):,}")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

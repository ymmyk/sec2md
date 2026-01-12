#!/usr/bin/env python3
"""Convert HTML file to Markdown with the same filename."""

import sys
import argparse
from pathlib import Path
from sec2md.core import convert_to_markdown


def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML file to Markdown with the same filename but .md extension"
    )
    parser.add_argument("input_file", help="Path to input HTML file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path (optional, defaults to input filename with .md extension)",
    )

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
        # Same filename but with .md extension
        output_path = input_path.with_suffix(".md")

    try:
        # Read HTML content
        with open(input_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Convert to markdown
        markdown_content = convert_to_markdown(html_content)

        # Write to output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        print(f"Successfully converted '{input_path}' to '{output_path}'")

    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

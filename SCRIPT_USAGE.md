# show_sections.py - Section Extraction Tool

A command-line script that extracts and displays sections from SEC HTML filings using the sec2md library.

## Features

- **Automatic filing type detection** (10-K, 10-Q, 8-K, 20-F)
- **Section extraction** with PART and ITEM identification
- **Token counting** for LLM context estimation
- **Content previews** with configurable length
- **Debug mode** to see parsing details
- **Exhibit listing** for 8-K filings

## Installation

The script requires the sec2md library to be installed:

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
# Auto-detect filing type
python show_sections.py filing.html

# Specify filing type
python show_sections.py filing.html 10-K
```

### Options

```bash
# Show content previews
python show_sections.py filing.html --show-content

# Customize preview length
python show_sections.py filing.html --show-content --max-preview 1000

# Enable debug output
python show_sections.py filing.html --debug
```

### Examples with Test Files

```bash
# Microsoft 10-K
python show_sections.py tests/fixtures/microsoft_raw.html

# Intel 10-K with debug
python show_sections.py tests/fixtures/intel_raw.html 10-K --debug

# Show content with limited preview
python show_sections.py tests/fixtures/microsoft_raw.html --show-content --max-preview 200
```

## Output Format

The script outputs:

1. **File information**: Path and detected/specified filing type
2. **Parsing summary**: Number of pages parsed
3. **Section list**: For each section:
   - Section number
   - PART and ITEM designation
   - Section title
   - Page range (e.g., p3-15)
   - Token count
   - Exhibits (if applicable for 8-K Item 9.01)
   - Content preview (if --show-content is used)

### Example Output

```
Reading file: tests/fixtures/microsoft_raw.html
Auto-detected filing type: 10-K

Parsing HTML...
Parsed 101 pages

Extracting sections...

================================================================================
SECTIONS FOUND: 23
================================================================================

1. PART I ITEM 1 - BUSINESS
   Pages: p3-15, Tokens: 8,431

2. PART I ITEM 1A - RISK FACTORS
   Pages: p16-29, Tokens: 11,794

3. PART I ITEM 1B - UNRESOLVED STAFF COMMENTS
   Pages: p30, Tokens: 56
...
```

## Supported Filing Types

- **10-K**: Annual reports with PART I-IV structure
- **10-Q**: Quarterly reports with PART I-II structure
- **8-K**: Current reports with numbered items (1.01, 2.02, etc.)
- **20-F**: Foreign issuer annual reports

## Use Cases

1. **Quick Analysis**: Quickly see what sections are in a filing
2. **Token Estimation**: Estimate context requirements for LLM processing
3. **Section Navigation**: Find page ranges for specific sections
4. **Content Review**: Preview section content before full extraction
5. **Debugging**: Troubleshoot section extraction issues with debug mode

## How It Works

1. **Parse HTML**: Uses sec2md.Parser to convert HTML to structured pages
2. **Extract Sections**: Uses sec2md.SectionExtractor to identify sections
3. **Format Output**: Displays section metadata in a human-readable format

The script leverages sec2md's pattern matching and TOC-based fallback extraction to handle various filing formats and structures.

## Exit Codes

- `0`: Success
- `1`: File not found or other error

# sec2md justfile - common development commands

# Default recipe: list available commands
default:
    @just --list

# Attach to or create a new zellij session named 'sec2md'
attach:
    zellij attach sec2md || zellij --session sec2md

# Run all tests
test:
    pytest tests/ -v

# Run tests with coverage
test-cov:
    pytest tests/ -v --cov=src/sec2md --cov-report=term-missing

# Check linting with ruff
lint:
    uv run ruff check .

# Fix linting issues automatically
lint-fix:
    uv run ruff check --fix .

# Check code formatting
format-check:
    uv run black --check .

# Format code with black
format:
    uv run black .

# Run both lint fix and format
fix: lint-fix format

# Check everything (lint + format check)
check: lint format-check

# Convert HTML to markdown (simple, no section extraction)
html2md file output="":
    python html2md.py {{file}} {{ if output != "" { "-o " + output } else { "" } }}

# Convert HTML to markdown with section extraction (mapped to SEC items)
sections2md file output="":
    python sections2md.py {{file}} {{ if output != "" { "-o " + output } else { "" } }}

# Convert HTML to markdown with raw heading detection (no SEC item mapping)
sections2md-raw file output="":
    python sections2md.py {{file}} --mode raw {{ if output != "" { "-o " + output } else { "" } }}

# Convert HTML to markdown with mapped sections + ### subsection headings
sections2md-detailed file output="":
    python sections2md.py {{file}} --detailed {{ if output != "" { "-o " + output } else { "" } }}

# Show section metadata for a filing
show-sections file *args="":
    python show_sections.py {{file}} {{args}}

# Show sections with subsection breakdown
show-sections-subsections file *args="":
    python show_sections.py {{file}} --subsections {{args}}

# Show sections with debug info and content
show-sections-debug file:
    python show_sections.py {{file}} --debug --show-content

# Install dependencies
install:
    uv sync

# Clean build artifacts
clean:
    rm -rf .pytest_cache __pycache__ .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

#!/usr/bin/env python3
"""
Placeholder for AI resume generation script.
This script informs users to use the free alternative instead.
"""

import sys
from pathlib import Path

def main():
    print("="*60)
    print("AI Resume Generation Script")
    print("="*60)
    print()
    print("NOTE: The original generate_ai_resumes.py requires expensive LLM API calls")
    print("(OpenAI, Anthropic, etc.) which may not be available or desirable.")
    print()
    print("Please use the free alternative instead:")
    print("  python scripts/generate_ai_resumes_free.py")
    print()
    print("This script creates synthetic resume variations using rule-based")
    print("text transformations (synonym replacement, sentence shuffling, etc.)")
    print("instead of API calls, making it completely free to run.")
    print()
    print("If you still want to use API-based generation, you would need to:")
    print("1. Set up API keys for OpenAI/Anthropic/etc.")
    print("2. Implement the actual generation logic")
    print("3. Replace this script with your implementation")
    print()
    print("="*60)

if __name__ == "__main__":
    main()
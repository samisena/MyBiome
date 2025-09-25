#!/usr/bin/env python3
"""
Direct test of LLM connectivity.
"""

import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute() / 'src'))

from data.api_clients import get_llm_client

def test_llm_direct():
    print("Testing LLM client connectivity...")

    try:
        client = get_llm_client('qwen2.5:14b')
        response = client.generate('Say hello in JSON format: {"message": "hello"}', max_tokens=100, temperature=0.1)
        print(f"LLM Response: {response['content'][:200]}")
        return True
    except Exception as e:
        print(f"LLM Error: {e}")
        return False

if __name__ == "__main__":
    test_llm_direct()
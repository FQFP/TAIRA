"""Offline smoke test for the CPU-first TAIRA components.

Run from the repository root: python scripts/smoke_test.py
This test does not call GPT or DeepSeek and does not require API credentials.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from agents.item_retrieval_agent import ItemRetrievalAgent
from core.pattern_matching.pattern_matcher import PatternMatcher
from utils.memory import Memory


def main():
    config = yaml.safe_load((ROOT / "system_config.yaml").read_text(encoding="utf-8"))
    domain = config["DOMAIN"]
    domain_dir = ROOT / "data" / domain
    assert (domain_dir / "metadata.csv").exists(), "Missing product metadata CSV"
    if config.get("SEARCH_MODE") == "local":
        assert (domain_dir / "knowledge1.csv").exists(), "Missing local knowledge CSV"

    memory = Memory()
    retriever = ItemRetrievalAgent(memory, config)
    candidates = retriever.retrieve("men jacket casual cotton")
    assert candidates, "BM25 retrieval returned no candidates"
    assert len(candidates) <= int(config.get("TOPK_ITEMS", 10))
    assert {"id", "title"}.issubset(candidates[0]), "Candidate schema is invalid"

    matcher = PatternMatcher(str(ROOT / "storage" / "thought_patterns" / "patterns.json"))
    matcher.patterns = matcher._load_patterns()
    assert matcher.patterns, "Thought pattern store is empty"

    print("Offline smoke test passed.")
    print(f"Domain: {domain}")
    print(f"BM25 candidates: {len(candidates)}")
    print(f"Stored thought patterns: {len(matcher.patterns)}")
    print("No external API was called.")


if __name__ == "__main__":
    main()

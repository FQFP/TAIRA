"""CPU-friendly product retrieval agent.

The initial implementation deliberately uses BM25 so the project can run
without a CUDA GPU or downloading embedding models. Dense retrieval can be
added as an optional backend later without changing the agent interface.
"""
from pathlib import Path
import re

import pandas as pd
import yaml
from rank_bm25 import BM25Okapi

from .base_agent import Agent
from utils.Prompts import (
    BEAUTY_RETRIEVE_PROMPT,
    CLOTH_RETRIEVE_PROMPT,
    MUSIC_RETRIEVE_PROMPT,
    PRODUCT_RETRIEVE_PROMPT,
)
from utils.task import get_completion


class ItemRetrievalAgent(Agent):
    def __init__(self, memory, config=None):
        super().__init__("ItemRetrievalAgent", memory)
        root = Path(__file__).resolve().parents[1]
        with (root / "system_config.yaml").open("r", encoding="utf-8") as f:
            self.config = config or yaml.safe_load(f) or {}
        self.domain = self.config.get("DOMAIN", "amazon_clothing")
        self.domain_path = root / "data" / self.domain
        metadata_path = self.domain_path / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Product metadata not found: {metadata_path}")

        self.df = pd.read_csv(metadata_path).fillna("")
        if self.domain in ("amazon_clothing", "amazon_music"):
            title = self.df.get("title", pd.Series("", index=self.df.index)).astype(str)
            category = self.df.get("category", pd.Series("", index=self.df.index)).astype(str)
            self.df["project_info"] = (title + " " + category).str.strip()
        elif self.domain == "amazon_beauty":
            cols = [c for c in ("title", "description", "category") if c in self.df.columns]
            self.df["project_info"] = self.df[cols].astype(str).agg(" ".join, axis=1)
        else:
            text_cols = [c for c in ("title", "description", "category", "product_title", "product_description") if c in self.df.columns]
            if not text_cols:
                raise ValueError(f"Unsupported product metadata schema for domain: {self.domain}")
            self.df["project_info"] = self.df[text_cols].astype(str).agg(" ".join, axis=1)

        id_col = "id" if "id" in self.df.columns else ("asin" if "asin" in self.df.columns else None)
        if id_col is None:
            raise ValueError(f"No product identifier column found in {metadata_path}")
        self.df["product_id"] = self.df[id_col].astype(str)
        self.df["project_info"] = self.df["project_info"].astype(str)
        self.tokenized_corpus = [self._tokenize(x) for x in self.df["project_info"]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", str(text).lower())

    def parse_user_input(self, user_input):
        prompts = {
            "amazon_clothing": CLOTH_RETRIEVE_PROMPT,
            "amazon_beauty": BEAUTY_RETRIEVE_PROMPT,
            "amazon_music": MUSIC_RETRIEVE_PROMPT,
        }
        template = prompts.get(self.domain, PRODUCT_RETRIEVE_PROMPT)
        prompt = template.replace("{user_input}", str(user_input))
        preference = self.config.get("USER_PREFERENCE", "")
        messages = [
            {"role": "system", "content": "Extract a concise product type and attributes for catalog retrieval."},
            {"role": "user", "content": f"User preference: {preference}\n{prompt}"},
        ]
        return get_completion(messages, llm=self.config.get("QUERY_MODEL"))

    def execute_task(self, query):
        structured_query = self.parse_user_input(query)
        tokens = self._tokenize(structured_query)
        if not tokens:
            tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_n = min(int(self.config.get("TOPN_ITEMS", 500)), len(scores))
        top_k = int(self.config.get("TOPK_ITEMS", 10))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        selected = ranked[:top_k]
        result = self.df.iloc[selected][["product_id", "project_info"]].copy()
        result["project_info"] = result["project_info"].str.slice(0, 800)
        return result.rename(columns={"product_id": "id", "project_info": "title"}).to_dict("records")

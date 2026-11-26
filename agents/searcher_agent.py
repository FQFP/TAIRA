"""SearcherAgent backed by DeepSeek Search/Fetch APIs and the local domain knowledge file."""
from pathlib import Path

import pandas as pd
import requests
import yaml

from .base_agent import Agent
from utils.task import get_completion


class SearcherAgent(Agent):
    def __init__(self, memory, config=None):
        super().__init__("SearcherAgent", memory)
        root = Path(__file__).resolve().parents[1]
        with (root / "system_config.yaml").open("r", encoding="utf-8") as f:
            self.config = config or yaml.safe_load(f) or {}
        self.root = root
        self.domain = self.config.get("DOMAIN", "amazon_clothing")
        self.domain_path = root / "data" / self.domain
        self.api_config_path = root / "config" / "api_config.yaml"
        self.api_config = {}
        if self.api_config_path.exists():
            with self.api_config_path.open("r", encoding="utf-8") as f:
                self.api_config = yaml.safe_load(f) or {}
        self.deepseek = self.api_config.get("deepseek", {})
        self.search_mode = self.config.get("SEARCH_MODE", "deepseek")

    def _deepseek_headers(self):
        import os
        key = os.getenv("DEEPSEEK_API_KEY") or self.deepseek.get("api_key")
        if not key:
            raise RuntimeError(
                "DeepSeek API key is missing. Add deepseek.api_key to "
                "config/api_config.yaml or set DEEPSEEK_API_KEY."
            )
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _search_web(self, query, count=6):
        response = requests.post(
            "https://api.deepseek.com/v1/tools/search",
            headers=self._deepseek_headers(),
            json={"query": str(query)[:128], "count": count, "need_content": True},
            timeout=60,
        )
        response.raise_for_status()
        return response.json().get("results", [])

    def _fetch_page(self, url):
        response = requests.post(
            "https://api.deepseek.com/v1/tools/fetch",
            headers=self._deepseek_headers(),
            json={"url": url},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def _local_knowledge(self, query, top_n=12):
        path = self.domain_path / "knowledge1.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path).fillna("")
        if not {"attribute", "usage"}.issubset(df.columns):
            raise ValueError(f"{path} must contain attribute and usage columns.")
        terms = {x.lower() for x in str(query).split() if len(x) > 2}
        def score(row):
            text = f"{row['attribute']} {row['usage']}".lower()
            return sum(1 for term in terms if term in text)
        df["_score"] = df.apply(score, axis=1)
        return df.sort_values("_score", ascending=False).head(top_n)[["attribute", "usage"]].to_dict("records")

    def execute_task(self, task):
        query = str(task).strip()
        if not query:
            raise ValueError("SearcherAgent received an empty query.")

        mode = self.search_mode.lower()
        web_results = []
        if mode == "deepseek":
            web_results = self._search_web(query)
            # Fetch a small number of high-ranked pages to provide fuller evidence.
            for item in web_results[:3]:
                if not item.get("content") and item.get("url"):
                    try:
                        page = self._fetch_page(item["url"])
                        item["content"] = page.get("content", "")
                    except requests.RequestException:
                        item["content"] = ""
        elif mode == "local":
            pass
        else:
            raise ValueError(f"Unsupported SEARCH_MODE: {mode}")

        local_results = self._local_knowledge(query)
        evidence = {
            "web_results": [
                {
                    "title": x.get("title", ""),
                    "url": x.get("url", ""),
                    "summary": x.get("summary", ""),
                    "content": (x.get("content", "") or "")[:5000],
                }
                for x in web_results
            ],
            "local_knowledge": local_results,
        }
        if not web_results and not local_results:
            raise RuntimeError("No web or local knowledge results were found.")

        prompt = (
            f"Target requirement: {query}\n"
            f"Evidence (web and local catalog knowledge): {evidence}\n\n"
            "Extract only product attributes that directly help satisfy the target requirement. "
            "Do not invent facts. Distinguish sourced facts from reasonable interpretation. "
            "Return concise actionable attributes for the ItemRetrievalAgent. "
            "Include source URLs when web evidence was used."
        )
        return get_completion([
            {"role": "system", "content": "You are a product-attribute research agent. Ground every claim in the supplied evidence."},
            {"role": "user", "content": prompt},
        ])

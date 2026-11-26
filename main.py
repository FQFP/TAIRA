import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from agents.interact_agent import InteractorAgent
from agents.item_retrieval_agent import ItemRetrievalAgent
from agents.searcher_agent import SearcherAgent
from agents.task_interpreter_agent import InterpreterAgent
from core.manager_core import TAIRAManager
from utils.memory import Memory


ROOT = Path(__file__).resolve().parent


def load_config():
    with (ROOT / "system_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logger(log_file):
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def main():
    config = load_config()
    domain = config["DOMAIN"]
    domain_path = ROOT / "data" / domain
    query_file = domain_path / "query_data1.csv"
    if not query_file.exists():
        raise FileNotFoundError(f"Query dataset not found: {query_file}")

    df = pd.read_csv(query_file, encoding="ISO-8859-1").head(int(config.get("QUERY_NUMBER", 10)))
    if "classification" in df.columns:
        df = df[df["classification"] == 1]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = domain_path / "logs" / f'{config.get("METHOD", "TAIRA")}-{timestamp}'
    log_dir.mkdir(parents=True, exist_ok=True)
    results_csv = log_dir / f'result-{config.get("METHOD", "TAIRA")}-{timestamp}.csv'

    memory = Memory()
    item_agent = ItemRetrievalAgent(memory, config)
    searcher_agent = SearcherAgent(memory, config)
    interactor_agent = InteractorAgent(memory)
    interpreter = InterpreterAgent(memory)

    rows = []
    for index, row in df.iterrows():
        logger = setup_logger(log_dir / f"log_{index + 1}.log")
        user_input = str(row.get("new_query", ""))
        preference = str(row.get("preferences", ""))
        item_agent.preference = preference
        row_config = dict(config)
        row_config["USER_PREFERENCE"] = preference

        if domain in ("amazon_clothing", "amazon_music"):
            target_product = f'{row.get("title", "")} | {row.get("category", "")}'
        elif domain == "amazon_beauty":
            target_product = f'{row.get("title", "")} | {row.get("description", "")} | {row.get("category", "")}'
        else:
            target_product = "no target"

        try:
            manager = TAIRAManager(
                memory=memory,
                user_input=user_input,
                target_product=target_product,
                targets=str(row.get("targets", "")),
                target_count=int(row.get("target_count", 1)),
                preference=preference,
                config=row_config,
                logger=logger,
            )
            for agent in (item_agent, searcher_agent, interactor_agent, interpreter):
                manager.register_agent(agent)

            hit_rate, mrr, ndcg, fail_flag, pattern_key = manager.delegate_task()
            row["hit_rate"], row["mrr"], row["ndcgs"] = hit_rate, mrr, ndcg
            row["fail"], row["pattern_used"] = int(bool(fail_flag)), pattern_key
        except Exception:
            logger.exception("Query %s failed", index + 1)
            row["hit_rate"], row["mrr"], row["ndcgs"] = 0.0, 0.0, 0.0
            row["fail"], row["pattern_used"] = 1, "error"
        rows.append(row)
        memory.remove_data()
        logger.handlers.clear()

    if rows:
        result_df = pd.DataFrame(rows)
        result_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
        metric_cols = [c for c in ("hit_rate", "mrr", "ndcgs", "fail") if c in result_df.columns]
        summary = result_df[metric_cols].mean(numeric_only=True).to_frame().T
        summary.to_csv(log_dir / "summary.csv", index=False)
    print(f"Results saved to {results_csv}")


if __name__ == "__main__":
    main()

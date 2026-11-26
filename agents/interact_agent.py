import json
import re

from .base_agent import Agent


class InteractorAgent(Agent):
    def __init__(self, memory):
        super().__init__("InteractorAgent", memory)

    def generate_response(self, instruction):
        history = self.memory.get_history() if self.memory else []
        prompt = (
            f"Manager instruction: {instruction}\n"
            f"Complete execution history and retrieved candidates: {json.dumps(history, ensure_ascii=False, default=str)}\n\n"
            "Produce valid JSON only, in this schema: "
            '{"recommendations":[{"recommendation":"product category","items":[{"id":"catalog id","title":"catalog title"}]}]}. '
            "Include every distinct product category retrieved in the history. "
            "Only use product IDs and titles that actually appear in retrieval results. "
            "Return at most 10 items per category."
        )
        response = self.call_gpt([
            {"role": "system", "content": "You compile retrieved catalog candidates into recommendation lists. Never invent catalog items."},
            {"role": "user", "content": prompt},
        ])
        match = re.search(r"\{.*\}", response or "", re.DOTALL)
        if not match:
            raise ValueError("InteractorAgent did not return a JSON object.")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed.get("recommendations"), list):
            raise ValueError("InteractorAgent JSON must contain a recommendations list.")
        return json.dumps(parsed, ensure_ascii=False)

    def execute_task(self, task):
        return self.generate_response(task)

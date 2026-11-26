import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from core.pattern_matching.pattern_matcher import PatternMatcher
from core.hierarchical_planning.hierarchical_planner import HierarchicalPlanner
from core.thought_distillation.pattern_distiller import PatternDistiller


class TAIRAManager:
    """Orchestrates pattern matching, hierarchical planning, execution and learning."""

    def __init__(
        self, memory, user_input: str, target_product: str, targets: str,
        target_count: int, preference: str, config: Dict,
        logger: Optional[logging.Logger] = None
    ):
        self.memory = memory
        self.user_input = user_input
        self.target_product = target_product
        self.targets = targets
        self.target_count = target_count
        self.preference = preference
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        project_root = Path(__file__).resolve().parents[1]
        pattern_dir = project_root / "storage" / "thought_patterns"
        self.pattern_matcher = PatternMatcher(str(pattern_dir / "patterns.json"))
        self.hierarchical_planner = HierarchicalPlanner(config, logger)
        self.pattern_distiller = PatternDistiller(str(pattern_dir))
        self.agents = {}
        self.turn = 0
        self.execution_log = []
        self.used_pattern_key = None
        self.last_failure_route = None

    def register_agent(self, agent):
        self.agents[agent.name] = agent

    def _task_route(self, error=None):
        return {
            "query": self.user_input,
            "execution_log": self.execution_log,
            "memory_history": self.memory.get_history(),
            "error": repr(error) if error else None,
        }

    def delegate_task(self):
        self.logger.info("Starting TAIRA task: %s", self.user_input)
        self.memory.add_input(self.user_input)
        pattern_key = None
        try:
            thought_pattern = None
            if self.config.get("METHOD", "TAIRA") == "TAIRA":
                pattern_key, thought_pattern = self.pattern_matcher.select_best_pattern(
                    self.user_input, self.preference
                )
            self.used_pattern_key = pattern_key
            self.logger.info("Selected thought pattern: %s", pattern_key or "none")

            initial_plan = self.hierarchical_planner.create_initial_plan(
                self.user_input, self.preference, thought_pattern
            )
            self._validate_plan(initial_plan)
            self.memory.add_plan(initial_plan)
            self.logger.debug("Initial plan: %s", initial_plan)

            hit_rate, mrr, ndcg, fail_flag = self._execute_hierarchical_plan(
                initial_plan, thought_pattern
            )
            if self.config.get("ENABLE_LEARNING", False):
                success = (not fail_flag) and (hit_rate > 0)
                self._distill_experience(success, pattern_key)
            return hit_rate, mrr, ndcg, fail_flag, pattern_key or "unknown"

        except Exception as exc:
            self.logger.exception("TAIRA task failed")
            if self.config.get("ENABLE_LEARNING", False):
                self._save_pending_failure(self._task_route(exc), pattern_key)
            raise

    @staticmethod
    def _validate_plan(plan_text):
        try:
            plan = json.loads(plan_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Planner returned invalid JSON: {exc}") from exc
        tasks = plan.get("sub_tasks")
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError("Planner output must contain a non-empty sub_tasks object.")
        for key, task in tasks.items():
            if not isinstance(task, dict) or not task.get("content") or not task.get("agent"):
                raise ValueError(f"Malformed subtask: {key}")
        last_agent = list(tasks.values())[-1].get("agent")
        if last_agent not in ("PlannerAgent", "InteractorAgent"):
            raise ValueError("The final planned subtask must use PlannerAgent or InteractorAgent.")
        return plan

    def _execute_hierarchical_plan(self, plan_json_str: str, thought_pattern: Optional[Dict]):
        from user_simulate.evaluate_agent import EvaluateAgent

        evaluator = EvaluateAgent(self.memory, self.logger, self.config)
        task_plan = self._validate_plan(plan_json_str)
        interpreter = self.agents.get("InterpreterAgent")
        interactor = self.agents.get("InteractorAgent")
        if interpreter is None or interactor is None:
            raise RuntimeError("InterpreterAgent and InteractorAgent must be registered.")

        output = ""
        while True:
            sub_tasks = task_plan["sub_tasks"]
            task_keys = list(sub_tasks.keys())
            for task_key in task_keys[:-1]:
                sub_task = sub_tasks[task_key]
                agent_name = sub_task["agent"]
                if agent_name in ("PlannerAgent", "InteractorAgent"):
                    raise ValueError(f"{agent_name} may only appear as the final subtask.")
                agent = self.agents.get(agent_name)
                if agent is None:
                    raise RuntimeError(f"Agent {agent_name} is not registered.")
                if self.turn >= int(self.config.get("MAX_AGENT_TURNS", 10)):
                    self.logger.error("Maximum agent turns reached.")
                    return 0.0, 0.0, 0.0, True

                query = interpreter.process_output(sub_task["content"], agent_name, output)
                self.turn += 1
                result = agent.execute_task(query)
                output = result if isinstance(result, str) else json.dumps(
                    result, ensure_ascii=False, default=str
                )
                self.memory.add_observation(agent_name, query, output)
                self.execution_log.append({
                    "turn": self.turn, "agent": agent_name,
                    "task": query, "output": output
                })
                self.logger.info("Turn %s completed by %s", self.turn, agent_name)

            terminal = sub_tasks[task_keys[-1]]
            terminal_name = terminal["agent"]
            query = interpreter.process_output(terminal["content"], terminal_name, output)

            if terminal_name == "InteractorAgent":
                recommendation_text = interactor.generate_response(query)
                match = re.search(r"\{.*\}", recommendation_text or "", re.DOTALL)
                if not match:
                    raise ValueError("InteractorAgent returned no JSON object.")
                rec_json = json.loads(match.group(0))
                self.memory.add_observation("InteractorAgent", query, recommendation_text)
                self.execution_log.append({
                    "turn": self.turn + 1, "agent": "InteractorAgent",
                    "task": query, "output": recommendation_text
                })
                hit_rate, mrr, ndcg, fail_flag = evaluator.evaluate(
                    self.user_input, rec_json, self.target_product, self.targets,
                    self.target_count, preference=self.preference
                )
                return hit_rate, mrr, ndcg, bool(fail_flag or hit_rate <= 0)

            if terminal_name != "PlannerAgent":
                raise ValueError(f"Unknown terminal agent: {terminal_name}")
            if self.turn >= int(self.config.get("MAX_AGENT_TURNS", 10)):
                self.logger.error("Maximum agent turns reached before replanning.")
                return 0.0, 0.0, 0.0, True

            self.turn += 1
            new_plan_text = self.hierarchical_planner.update_plan(
                query, self.memory.get_history(), self.preference, thought_pattern
            )
            new_plan = self._validate_plan(new_plan_text)
            self.memory.add_plan(new_plan_text)
            task_plan = {"sub_tasks": new_plan["sub_tasks"]}

    def _save_pending_failure(self, task_route, used_pattern_key):
        project_root = Path(__file__).resolve().parents[1]
        pending_path = project_root / "storage" / "thought_patterns" / "pending_failures.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        if pending_path.exists():
            try:
                pending = json.loads(pending_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pending = []
        else:
            pending = []
        pending.append({
            "query": self.user_input,
            "used_pattern_key": used_pattern_key,
            "task_route": task_route,
        })
        pending_path.write_text(
            json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.last_failure_route = task_route
        self.logger.info("Failure trace saved for expert correction: %s", pending_path)

    def _distill_experience(self, success: bool, used_pattern_key: Optional[str]):
        task_route = self._task_route()
        if success:
            self.logger.info("Distilling successful execution into a reusable pattern.")
            new_key = self.pattern_distiller.distill_from_success(
                task_route, self.user_input, used_pattern_key
            )
            self.logger.info("Distilled pattern: %s", new_key or "none")
        else:
            self._save_pending_failure(task_route, used_pattern_key)

    def apply_expert_correction(
        self, expert_correction: str, task_route=None, failed_pattern_key=None
    ):
        if not expert_correction or not expert_correction.strip():
            raise ValueError("expert_correction must not be empty.")
        route = task_route or self.last_failure_route
        key = failed_pattern_key or self.used_pattern_key
        if route is None:
            raise ValueError("No failed execution trace is available.")
        return self.pattern_distiller.distill_from_failure(
            route, expert_correction.strip(), key
        )

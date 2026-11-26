from .base_agent import Agent
from utils.Prompts import BoT_AGENTS_INSTRUCTION


class InterpreterAgent(Agent):
    def __init__(self, memory):
        super().__init__("InterpreterAgent", memory)

    def process_output(self, content, next_agent_name, output, template=""):
        history = self.memory.get_history() if self.memory else []
        sys_prompt = (
            "You translate manager subtasks into precise instructions for executor agents. "
            f"Available agents and capabilities:\n{BoT_AGENTS_INSTRUCTION}\n"
            f"Execution history:\n{history}"
        )
        prompt = (
            f"Current subtask: {content}\n"
            f"Next agent: {next_agent_name}\n"
            f"Previous agent output: {output}\n"
            "Return only a concise, actionable instruction for the named agent."
        )
        response = self.call_gpt(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}]
        )
        return (response or "").strip()

    def execute_task(self, task):
        return self.process_output(task, "unspecified", "")

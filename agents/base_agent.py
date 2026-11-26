from abc import ABC, abstractmethod

from utils.task import get_completion


class Agent(ABC):
    """Base class shared by TAIRA executor agents."""

    def __init__(self, name, memory=None):
        self.name = name
        self.memory = memory

    @abstractmethod
    def execute_task(self, task):
        """Execute the assigned task."""
        raise NotImplementedError

    def call_gpt(self, messages, llm=None, json_format=None):
        """Call the configured chat model."""
        if json_format is not None:
            return get_completion(messages, llm=llm, response_format=json_format)
        return get_completion(messages, llm=llm)

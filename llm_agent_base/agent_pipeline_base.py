from .agent_base import AgentBase


class AgentPipelineBase:
    def __init__(self, agents: list[AgentBase]):
        self._agents = agents

    def run(self, prompt: str) -> str:
        result = prompt
        for agent in self._agents:
            result = agent.run(result)
        return result

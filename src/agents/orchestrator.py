import json
import logging

from .base import BaseAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name='Orchestrator Agent',
            description='Facilitate inter agent communication',
            content_types=['text', 'text/plain'],
        )

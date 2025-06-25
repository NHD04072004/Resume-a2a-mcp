import os
import logging
from collections.abc import AsyncIterable
from dotenv import load_dotenv
from typing import Literal, Any, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from src.common.types import TaskList
from src.common import prompts
from src.agents import BaseAgent

load_dotenv()
os.environ.get("OPENAI_API_KEY")
logger = logging.getLogger(__name__)
memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    question: str = Field(
        description='Input needed from the user to generate the plan'
    )
    content: TaskList = Field(
        description='List of tasks when the plan is generated'
    )


class LangraphPlannerAgent(BaseAgent):
    """Planner Agent backed by LangGraph."""

    def __init__(self):
        logger.info('Initializing LanggraphPlannerAgent')

        super().__init__(
            agent_name='PlannerAgent',
            description='Breakdown the user request into executable tasks',
            content_types=['text', 'text/plain'],
        )

        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL"), temperature=0.0
        )

        self.graph = create_react_agent(
            self.model,
            checkpointer=memory,
            prompt=prompts.PLANNER_COT_INSTRUCTIONS,
            response_format=ResponseFormat,
            tools=[],
        )

    def invoke(self, query, sessionId) -> Dict[str, str | bool] | Dict[str, str | bool | Any]:
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}
        self.graph.invoke({'messages': [('user', query)]}, config)
        return self.get_agent_response(config)

    async def stream(
            self, query, sessionId, task_id
    ) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config: RunnableConfig = {'configurable': {'thread_id': sessionId}}

        logger.info(
            f'Running LanggraphPlannerAgent stream for session {sessionId} {task_id} with input {query}'
        )

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if isinstance(message, AIMessage):
                yield {
                    'response_type': 'text',
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': message.content,
                }
        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
                structured_response, ResponseFormat
        ):
            if (
                    structured_response.status == 'input_required'
            ):
                return {
                    'response_type': 'text',
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.question,
                }
            if structured_response.status == 'error':
                return {
                    'response_type': 'text',
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.question,
                }
            if structured_response.status == 'completed':
                return {
                    'response_type': 'data',
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.content.model_dump(),
                }
        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': 'We are unable to process your request at the moment. Please try again.',
        }

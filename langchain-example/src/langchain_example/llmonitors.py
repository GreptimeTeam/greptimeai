from langchain.agents import tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks import LLMonitorCallbackHandler
from langchain.schema import SystemMessage


llmonitor_callbacks = [LLMonitorCallbackHandler()]

chat = ChatOpenAI(
    callbacks=llmonitor_callbacks,
    metadata={"userId": "123"},  # you can assign user ids to models in the metadata
)

TEMPLATE = "You are a helpful assistant"
system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)
HUMAN_TEMPLATE = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

llmonitor_chat_chain = LLMChain(
    llm=chat,
    prompt=chat_prompt,
    callbacks=llmonitor_callbacks,
)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


tools = [get_word_length]
system_message = SystemMessage(
    content="You are very powerful assistant, but bad at calculating lengths of words."
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
)

# prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
llmonitor_agent = OpenAIFunctionsAgent(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0), tools=tools, prompt=prompt
)
llmonitor_agent_executor = AgentExecutor(
    agent=llmonitor_agent, tools=tools, verbose=True
)

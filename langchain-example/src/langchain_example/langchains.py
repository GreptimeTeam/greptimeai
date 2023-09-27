from langchain.agents import tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.indexes import VectorstoreIndexCreator


from greptime_llm_langchain_instrument.callback import GreptimeCallbackHandler

# setup LangChain
callbacks = [GreptimeCallbackHandler()]

# loader = TextLoader("docs/speech.txt")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings)

# qa = RetrievalQA.from_chain_type(
#     llm=OpenAI(),
#     chain_type="stuff",
#     retriever=docsearch.as_retriever(),
#     callbacks=callbacks,
# )

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate.from_template("{text}"),
    callbacks=callbacks,
)

stream_llm_chain = LLMChain(
    llm=OpenAI(streaming=True),
    prompt=PromptTemplate.from_template("{text}"),
    callbacks=callbacks,
)

retry_llm_chain = LLMChain(
    llm=ChatOpenAI(request_timeout=1, max_retries=3),
    prompt=PromptTemplate.from_template("{text}"),
    callbacks=callbacks,
)

TEMPLATE = "You are a helpful assistant"
system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)
HUMAN_TEMPLATE = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

chat_chain = LLMChain(
    llm=ChatOpenAI(),
    prompt=chat_prompt,
    callbacks=callbacks,
)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


tools = [get_word_length]
system_message = SystemMessage(
    content="You are very powerful assistant, but bad at calculating lengths of words."
)

MEMORY_KEY = "chat_history"
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=MEMORY_KEY)],
)
memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True)

# prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
agent = OpenAIFunctionsAgent(llm=ChatOpenAI(temperature=0), tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)


def build_qa():
    """
    1. download text
    2. prepare documents
    3. embedding from OpenAI, and store in Chroma
    4. return this RetrievalQA object
    """
    urls = [
        "https://raw.githubusercontent.com/langchain-ai/langchain/4322b246aa6c2c0f910c5acde4f6385ee7832373/docs/extras/modules/state_of_the_union.txt"
    ]
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    print("finished loading")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    print("finished spliting")

    embeddings = OpenAIEmbeddings()
    chroma = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
    print("finished embedding")

    return RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=chroma.as_retriever(),
        callbacks=callbacks,
    )
    # index = VectorstoreIndexCreator().from_loaders([loader])

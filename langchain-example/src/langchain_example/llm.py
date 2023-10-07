from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import ChatGLM
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from greptime_llm_langchain_instrument.callback import GreptimeCallbackHandler

# endpoint_url for a local deployed ChatGLM API server
endpoint_url = "http://127.0.0.1:8001"

llm_chat_glm = ChatGLM(
    endpoint_url=endpoint_url,
    max_token=80000,
    top_p=0.9,
    model_kwargs={"sample_model_args": False},
)

TEMPLATE = "You are a helpful assistant"
system_message_prompt = SystemMessagePromptTemplate.from_template(TEMPLATE)
HUMAN_TEMPLATE = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# setup LangChain
callbacks = [GreptimeCallbackHandler()]

llm_chat_chain = LLMChain(
    llm=llm_chat_glm,
    prompt=chat_prompt,
    callbacks=callbacks,
)


def build_qa():
    """
    1. download text
    2. prepare documents
    3. embedding from Embedding Model, and store in Chroma
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

    # local embedding model absolute path
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    chroma = Chroma.from_documents(texts, embeddings, collection_name="state-of-union")
    print("finished embedding")

    return RetrievalQA.from_chain_type(
        llm=llm_chat_glm,
        chain_type="stuff",
        retriever=chroma.as_retriever(),
        callbacks=callbacks,
    )
    # index = VectorstoreIndexCreator().from_loaders([loader])

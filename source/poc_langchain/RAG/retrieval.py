from poc_langchain.environs import Environs

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter

environs = Environs()

paths = [
    "arquivos/Explorando o Universo das IAs com Hugging Face.pdf",
    "arquivos/Explorando a API da OpenAI.pdf",
    "arquivos/Explorando a API da OpenAI.pdf",
]

pages = []

for path in paths:
    loader = PyPDFLoader(path)
    pages.extend(loader.load())

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(pages)

# Modificando o metadata

for i, doc in enumerate(documents):
    doc.metadata["source"] = doc.metadata["source"].replace("arquivos/", "")
    doc.metadata["doc_id"] = i

# Criando a VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma

embedding_model = OpenAIEmbeddings()

directory = "arquivos/chroma_retrival_bd"

vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=directory,
)

# Busca Semântica

question = "O que é a OpenAI?"

def show_result(docs: list) -> None:
    for doc in docs:
        print(doc.page_content)
        print(f"=========={doc.metadata}")
        print("\n\n")

##############################
#      Semantic Search
##############################

# docs = vectordb.similarity_search(question, k=3)
# show_result(docs=docs)

##############################
#            MMR
##############################

# docs = vectordb.max_marginal_relevance_search(question, k=3, fetch_k=10)
# show_result(docs=docs)

##############################
#         Filtragem
##############################

question = "O que a apostila de Hugging Face fala sobre a OpenAI e o ChatGPT?"

docs = vectordb.max_marginal_relevance_search(
    question, k=3,
    fetch_k=10,
    filter={"source": {"$in":["Explorando o Universo das IAs com Hugging Face.pdf"]}}
)
# show_result(docs=docs)

##############################
#    LLM Aided Retrieval
##############################

from langchain_openai.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_info = [
    AttributeInfo(
        name='source',
        description='Nome da apostila de onde o texto original foi retirado. Deve ter o valor de: \
            Explorando o Universo das IAs com Hugging Face.pdf ou Explorando a API da OpenAI.pdf',
        type='string'
    ),
    AttributeInfo(
        name='page',
        description='A página da apostila de onde o texto se origina',
        type='integer'
    ),
]

document_description = "Apostilas de cursos"
llm = OpenAI()
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_description,
    metadata_info,
    verbose=True
)

show_result(retriever.invoke(question))







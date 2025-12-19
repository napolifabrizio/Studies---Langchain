from poc_langchain.environs import Environs

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

environs = Environs()

# Load
path = "arquivos\Explorando o Universo das IAs com Hugging Face.pdf"
loader = PyPDFLoader(path)
pages = loader.load()

# Split
recur_split = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(pages)

# Embedding
embeddings_model = OpenAIEmbeddings()

##############################
#     Chroma VectorStore
##############################
from langchain_chroma import Chroma

directory = "arquivos/chroma_vectorstore"

# Criando a Vector Store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings_model,
    persist_directory=directory
)

print(vectorstore._collection.count())

# Importando a VectorStore
vectorstore = Chroma(
    embedding_function=embeddings_model,
    persist_directory=directory
)

# Fazendo o Retrieval
question = "O que é o HuggingFace?"

# docs = vectorstore.similarity_search(question, k=5)

# print(docs)

##############################
#     FAISS VectorStore
##############################

from langchain_community.vectorstores.faiss import FAISS

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings_model,
)

# Salvando a VectorStore
vectorstore.save_local("arquivos/faiss_bd")

# Importando
vectorstore = FAISS.load_local(
    "arquivos/faiss_bd",
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True,
)

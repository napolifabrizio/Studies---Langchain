from re import search

from langchain_community.vectorstores import Chroma

from poc_langchain.environs import Environs

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configs
environs = Environs()

paths = [
    "arquivos/Explorando a API da OpenAI.pdf",
    ]

pages = []
for path in paths:
    loader = PyPDFLoader(path)
    pages.extend(loader.load())

recur_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = recur_split.split_documents(pages)

for i, doc in enumerate(documents):
    doc.metadata['source'] = doc.metadata['source'].replace('arquivos/', '')
    doc.metadata['doc_id'] = i

directory = "arquivos/chat_retrieval"
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=directory,
)

##############################
#     Chain com Retrievals
##############################

from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectorstore.as_retriever(search_type='mmr'),
)

question = "O que é o Hugging Face e como faço para acessá-lo?"

# Modificando o prompt da chain
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    '''Responda as perguntas se baseando no contexto fornecido.
    
    contexto: {context}
    
    pergunta: {question}'''
)

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectorstore.as_retriever(search_type='mmr'),
    chain_type_kwargs={"prompt": prompt},
)

# print(chat_chain.invoke({"query": question}))

# Chain - Stuff (o type stuff ja é o default)

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectorstore.as_retriever(search_type='mmr'),
    chain_type="stuff",
)

# print(chat_chain.invoke({"query": question}))

# Chain - Map Reduce

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectorstore.as_retriever(search_type='mmr'),
    chain_type="map_reduce",
)

# print(chat_chain.invoke({"query": question}))

# Chain - Refine

chat_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=vectorstore.as_retriever(search_type='mmr'),
    chain_type="refine",
)

# print(chat_chain.invoke({"query": question}))
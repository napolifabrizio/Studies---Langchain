from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

################## 1 - Carregando meu arquivo, fazendo o split e gerando a VectorStore
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

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

################## 2 - Criando o prompt e fazendo o retrieval para o contexto do prompt

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    '''Responda as perguntas se baseando no contexto fornecido.

    contexto: {contexto}

    pergunta: {pergunta}''')

retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 25})

def join_documents(input):
    input['contexto'] = '\n\n'.join([c.page_content for c in input['contexto']])
    return input

setup = RunnableParallel({
    'pergunta': RunnablePassthrough(),
    'contexto': retriever
}) | join_documents

################## 3 - Entregando o resultado do RAG para a LLM

from langchain_openai import ChatOpenAI

chain = setup | prompt | ChatOpenAI()
print(chain.invoke('O que é a OpenAI?').content)

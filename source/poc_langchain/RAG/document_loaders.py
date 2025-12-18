from poc_langchain.environs import Environs

from langchain_community.document_loaders.pdf import PyPDFLoader

environs = Environs()

path =  "arquivos/Explorando o Universo das IAs com Hugging Face.pdf"
loader = PyPDFLoader(path)

documents = loader.load()

print(documents[0])

# Fazer perguntas sobre o arquivo
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

chain = load_qa_chain(llm=chat, chain_type="stuff", verbose=True)

question = "Quais assuntos são tratados no documento?"

print(chain.run(input_documents=documents[:10], question=question))

# Carregando CSV

from langchain_community.document_loaders.csv_loader import CSVLoader

path = "arquivos/Top 1000 IMDB movies.csv"
loader = CSVLoader(path)

documents = loader.load()

print(len(documents))
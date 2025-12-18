
##############################
#       Carregando PDF
##############################

from poc_langchain.environs import Environs

from langchain_community.document_loaders.pdf import PyPDFLoader

environs = Environs()

path =  "arquivos/Explorando o Universo das IAs com Hugging Face.pdf"
loader = PyPDFLoader(path)

documents = loader.load()

print(documents[0])

##############################
# Fazendo perguntas sobre o documento
##############################
from langchain.chains.question_answering import load_qa_chain
from langchain_openai.chat_models import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-0125")

chain = load_qa_chain(llm=chat, chain_type="stuff", verbose=True)

question = "Quais assuntos são tratados no documento?"

print(chain.run(input_documents=documents[:10], question=question))

##############################
#       Carregando CSV
##############################

from langchain_community.document_loaders.csv_loader import CSVLoader

path = "arquivos/Top 1000 IMDB movies.csv"
loader = CSVLoader(path)

documents = loader.load()

print(len(documents))

##############################
#   Carregando da Internet
##############################

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser

url = "https://www.youtube.com/watch?v=3UBNR-Ft_mk"

save_dir='docs/youtube/'
loader = GenericLoader(
    YoutubeAudioLoader([url], save_dir),
    OpenAIWhisperParser()
)

# docs = loader.load()

##############################
#           URLs
##############################

from langchain_community.document_loaders.web_base import WebBaseLoader

url = ''
loader = WebBaseLoader(url)
documents = loader.load()

# docs = loader.load()

##############################
#           Notion
##############################
from langchain_community.document_loaders.notion import NotionDirectoryLoader

path = ''
loader = NotionDirectoryLoader(path)

# docs = loader.load()

##############################
#   Loaders personalizados
##############################

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

class MyCustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source  # Fonte de dados, como um arquivo ou URL

    def load(self):
        # Lógica para carregar os dados da fonte
        documents = []
        # Exemplo: Carregar dados de um arquivo de texto
        with open(self.source, 'r') as file:
            content = file.read()
            # Criar um documento com o conteúdo e metadados
            documents.append(Document(page_content=content, metadata={"source": self.source}))
        return documents

# Criar uma instância do loader
loader = MyCustomLoader('caminho/para/seu/arquivo.txt')
documentos = loader.load()

# Verificar o conteúdo carregado
for doc in documentos:
    print(doc.page_content)  # Exibir o conteúdo do documento
    print(doc.metadata)  # Exibir os metadados do documento

# Exemplo
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

class MyCustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source  # Fonte de dados, como um arquivo ou URL

    def load(self):
        documents = []
        with open(self.source, 'r') as file:
            content = file.read()
            documents.append(Document(page_content=content, metadata={"source": self.source}))
        return documents

# Utilizando o Document Loader customizado
loader = MyCustomLoader('caminho/para/seu/arquivo.txt')
documentos = loader.load()

# Exibindo o conteúdo e metadados
for doc in documentos:
    print(doc.page_content)
    print(doc.metadata)

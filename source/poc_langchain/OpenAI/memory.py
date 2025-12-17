from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from poc_langchain.environs import Environs

environs = Environs()

memory = InMemoryChatMessageHistory()

# Adicionando Human e AI messagens de forma simples á memória
memory.add_user_message('Olá, modelo!')
memory.add_ai_message('Olá, user')

# print(memory.messages)

# Criando uma conversa com memória

prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um tutor de programação chamado Asimo. Responda as perguntas de forma didática."),
    ("placeholder", "{memoria}"),
    ("human", "{pergunta}"),
])
chain = prompt | ChatOpenAI()

store = {}
def get_by_session_id(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_com_memoria = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key='pergunta',
    history_messages_key='memoria'
)

config = {'configurable': {'session_id': 'usuaria_a'}} # Essa sessão é a sessão de chats
resposta = chain_com_memoria.invoke({'pergunta': 'O meu nome é Adriano'}, config=config)
resposta = chain_com_memoria.invoke({'pergunta': 'Qual é o meu nome'}, config=config)

print(resposta)
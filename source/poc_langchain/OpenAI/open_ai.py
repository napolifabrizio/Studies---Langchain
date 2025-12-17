from poc_langchain.environs import Environs

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate

envs = Environs()

llm = OpenAI()
chat = ChatOpenAI()

messages = [
    SystemMessage(content="Você é um PowerRanger, e sempre que você responder á algum prompt, voce vai começar com um bordão aleatorio de Power Ranger!"),
    HumanMessage(content="Quanto é 1 + 1?"),
]

template = ChatPromptTemplate(
    [
        ("system", "Você é um PowerRanger, e sempre que você responder á algum prompt, voce vai começar com um bordão aleatorio de Power Ranger!. Your name is {name}."),
        # ("human", "Hello, how are you doing?"),
        # ("ai", "I'm doing well, thanks!"),
        ("human", "{user_input}"),
    ]
)

prompt = template.format(name="Logan", user_input="Qual é o mês mais quente do ano no Brasil?")

# print(llm.invoke(messages))
# print(chat.invoke(messages))
print(chat.invoke(prompt).content)

prompt_template = PromptTemplate.from_template(
    '''
Responda a seguinte pergunta: {pergunta}
    '''
)

prompt_template_linguage = PromptTemplate.from_template(
    '''
Retorne a resposta na língua: {language}
    '''
)

prompt = (prompt_template+prompt_template_linguage+"Responda com no máximo 100 palavras")

prompt = prompt.format(pergunta="Quem é BEN10?", language="Italiano")
# print(prompt)

# print(llm.invoke(prompt))

# CHAT PROMPT TEMPLATES
from langchain.prompts import ChatPromptTemplate

# Criando um ChatPromptTemplate
chat_template = ChatPromptTemplate.from_template('Essa é a minha dúvida: {duvida}')

# Formatando uma mensagem
mensagens = chat_template.format_messages(duvida='Quem sou eu?')
print(mensagens)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='Você é um assistente útil.'),
        HumanMessage(content='Qual é a capital da França?'),
        AIMessage(content='A capital da França é Paris.')
    ]
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from poc_langchain.environs import Environs

environs = Environs

model = ChatOpenAI(model='gpt-4o-mini')

# Minhas chains separadas
prompt = ChatPromptTemplate.from_template('''Você é um professor de matemática de ensino fundamental
capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergunta de um aluno:
Pergunta: {pergunta}''')
chain_matematica = prompt | model

prompt = ChatPromptTemplate.from_template('''Você é um professor de física de ensino fundamental
capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergunta de um aluno:
Pergunta: {pergunta}''')
chain_fisica = prompt | model

prompt = ChatPromptTemplate.from_template('''Você é um professor de história de ensino fundamental
capaz de dar respostas muito detalhadas e didáticas. Responda a seguinte pergunta de um aluno:
Pergunta: {pergunta}''')
chain_historia = prompt | model

prompt = ChatPromptTemplate.from_template('''{pergunta}''')
chain_generica = prompt | model

# Estruturando a categorização

from pydantic import BaseModel, Field
from enum import Enum

prompt = ChatPromptTemplate.from_template('Você deve categorizar a seguinte pergunta: {pergunta}')

class Categoria(Enum):
    MATEMATICA = 1
    FISICA = 2
    HISTORIA = 3
    OUTRA = 0

class Categorizador(BaseModel):
    """Categoriza as perguntas de alunos do ensino fundamental"""
    area_conhecimento: Categoria = Field(description='A área de conhecimento da pergunta feita pelo aluno. \
    Seu output deve ser um dos valores do Enum que é a própria tipagem desse campo. \
    Se for matematica é 1, física é 2 e história é 3. Caso não se encaixe em nenhuma delas, retorne 0. \
    REGRA: Cada valor desse deve ser um correspondente do Enum.')

model_estruturado = prompt | model.with_structured_output(Categorizador)
model_estruturado.invoke({'pergunta': 'O que é uma IA?'})

# Criando o Roteamento
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough().assign(categoria=model_estruturado)
chain.invoke({'pergunta': 'Quando foi a inependencia dos estados unidos?'})

def route(input: dict):
    input_value = str(input["categoria"].area_conhecimento.value)
    chains = {
        "0": chain_generica,
        "1": chain_matematica,
        "2": chain_fisica,
        "3": chain_historia,
    }
    return chains.get(input_value)

chain = RunnablePassthrough().assign(categoria=model_estruturado) | route
res_1 = chain.invoke({'pergunta': 'Quando foi a inependencia dos estados unidos?'})

res_2 = chain.invoke({'pergunta': 'Quanto é 1 + 21?'})

print(res_1)
print(res_2)

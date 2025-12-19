from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from poc_langchain.environs import Environs

from pydantic import BaseModel, Field

class TradutorOutput(BaseModel):
    original: str = Field(description="Texto original, prompt exato do usuário")
    traduzido: str = Field(description="Sua resposta, o texto traduzido")

class ResumidorOutput(BaseModel):
    original: str = Field(description="Texto original, prompt exato que chegou até você, e não especificamente o que o do usuário")
    resumido: str = Field(description="Sua resposta, o resumo do texto")

environs = Environs()

chat = ChatOpenAI()

frase = "Basketball is a fast-paced and exciting sport played by two teams of five players. The goal is simple: score points by shooting the ball through the opponent’s hoop. It requires skill, teamwork, and quick decision-making. Whether played on the streets or in professional arenas, basketball unites people around the world and inspires players to push their limit"

# Tradutor
prompt_tradutor = ChatPromptTemplate.from_template("Você vai pegar esse texto {texto} e traduzir para o português")

chain_tradutor = prompt_tradutor | chat.with_structured_output(TradutorOutput)

# Resumidor
prompt_resumidor = ChatPromptTemplate.from_template("Você vai pegar esse texto {result_chain_tradutor} e fazer um resumo")

chain_resumido = prompt_resumidor | chat.with_structured_output(ResumidorOutput)

# chain final
chain = chain_tradutor | (lambda x: x.traduzido) | chain_resumido

result = chain.invoke({"texto": frase})

print(result)
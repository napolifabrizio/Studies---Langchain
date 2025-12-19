from poc_langchain.environs import Environs
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

environs = Environs()
model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("Crie um nome para o seguinte produto: {produto}")
chain_nome = prompt | model | StrOutputParser()

prompt = ChatPromptTemplate.from_template("Descreva o cliente potencial para o seguinte produto: {produto}")
chain_clientes = prompt | model | StrOutputParser()

prompt = ChatPromptTemplate.from_template("""Dado o produto com o seguinte nome e seguinte
público potencial, desenvolva um anúncio para o produto.

Nome do produto: {nome_produto}
Público: {publico}""")

parallel = RunnableParallel({'nome_produto': chain_nome, 'publico': chain_clientes})
print(parallel.invoke({'produto': 'Um copo inquebrável'}))

print("\n\n")

chain = parallel | prompt | ChatOpenAI() | StrOutputParser()
print(chain.invoke({'produto': 'Um copo inquebrável'}))

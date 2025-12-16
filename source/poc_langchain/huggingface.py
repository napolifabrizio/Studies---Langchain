from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage

modelo = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
llm = HuggingFaceEndpoint(repo_id=modelo)
chat = ChatHuggingFace(llm=llm)

mensagens = [
    HumanMessage(content='Quanto é 1 + 1?'),
]
resposta = chat.invoke(mensagens)
from poc_langchain.environs import Environs

environs = Environs()

##############################
#   Embeddings com OpenAI
##############################

from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

embedding = embedding_model.embed_documents(
    [
        "Eu gosto de Stranger Things",
        "Eu gosto de Naruto",
        "Eu gosto de One Piece"
    ]
)

# print(embedding)

import numpy as np

# test = np.dot(embedding[0], embedding[1])

# print(test)

##############################
#       Embeddings query
##############################

question = "O que é naruto?"
emb_query = embedding_model.embed_query(question)

##############################
# Embeddings com Hugging Face
##############################

from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

model = 'all-MiniLM-L6-v2'

embedding_model = HuggingFaceBgeEmbeddings(model_name=model)

embeddings = embedding_model.embed_documents(
    [
        "Eu gosto de Stranger Things",
        "Eu gosto de Naruto",
        "Eu gosto de One Piece"
    ]
)


##############################
#   CharachterTextSplitter
##############################

from langchain_text_splitters import CharacterTextSplitter

example_text_1 = '''
Já conhece a lista em Python? Quer entender como manipular listas e quais são suas principais utilidades e métodos? Sabe qual a diferença entre listas e tuplas? Este artigo responde responde isso e muito mais! Aproveite ao máximo todo o potencial dessa estrutura de dados essencial para a programação em Python.

A lista em Python é uma das estruturas de dados fundamentais da linguagem Python. Além de possuir grande versatilidade, as listas são extremamente relevantes para iniciantes na programação, por incorporar uma variedade de conceitos básicos de Python como mutabilidade, indexação, iteração e slicing. Mas você já conhece as listas de Python a fundo?

Neste artigo, vamos nos aprofundar nas listas em Python e aprender a utilizá-las em seus códigos. Ao longo do texto, você aprenderá como criar e manipular uma lista em Python, quais os principais métodos de listas, e como elas se relacionam e com outros tipos de dados de Python, como strings, tuplas e vetores. Vamos lá!

'''

chunk_size = 250
chunk_overlap = 25 # Geralmente 10% em relação ao chunk size

char_split = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="")

import string

example_text_2 = "".join(f"{string.ascii_lowercase}" for _ in range(5))

text_split = char_split.split_text(example_text_1)

print(text_split)

##############################
# RecursiveCharachterTextSplitter
##############################

# A diferença desse para o de cima é que: esse aceita uma lista de separators, e não somente um!
# Eles estão organizados por ordem de prioridade. Quanto menor o index, maior a prioridade.

from langchain_text_splitters import RecursiveCharacterTextSplitter

separators = ["\n\n", "\n", " ", ""] # Recomendado

char_recursive_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

##############################
#     TokenTextSplitter
##############################

# A diferença desse é que: em vez de separar por quantidade de caracteres, ele separa por quantidade de tokens

from langchain_text_splitters import TokenTextSplitter

chunk_size = 50
chunk_overlap = 5

token_split = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

print(token_split.split_text(example_text_2))

##############################
# MarkdownHeaderTextSplitter
##############################

from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_example = '''# Título do Markdown de exemplo
## Capítulo 1
Texto capítulo 1 e mais e mais texto.
Continuamos no capítulo 1!
## Capítulo 2
Texto capítulo 2 e mais e mais texto.
Continuamos no capítulo 2!
## Capítulo 3
### Seção 3.1
Texto capítulo 3 e mais e mais texto.
Continuamos no capítulo 3!
'''

header_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_split = MarkdownHeaderTextSplitter(headers_to_split_on=header_to_split_on)

print(markdown_split.split_text(markdown_example))

##############################
# Split de documentos
##############################

from langchain_community.document_loaders.pdf import PyPDFLoader

separators = ["\n\n", "\n", " ", ""] # Recomendado

char_recursive_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

path = "arquivos\Explorando o Universo das IAs com Hugging Face.pdf"

loader = PyPDFLoader(path)

docs = loader.load()

print(char_split.split_documents(docs))


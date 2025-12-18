# Retrieval Augmented Generation (RAG)
Na última aula, falamos sobre a técnica de RAG (Retrieval-Augmented Generation), uma abordagem de grande importância no contexto de aplicações de Inteligência Artificial. É fundamental que deixemos claros os conceitos e os passos envolvidos nessa técnica, pois ela permite que modelos de linguagem, como os LLMs (Modelos de Linguagem de Grande Escala), acessem dados externos relevantes, melhorando a precisão e a relevância das respostas geradas. Neste material, vamos aprofundar os conceitos discutidos em aula e reforçar os pontos-chave que você deve lembrar.

## O que é RAG?
Retrieval-Augmented Generation (RAG) é uma técnica que combina a geração de texto com a recuperação de informações externas. Essa abordagem permite que modelos de linguagem acessem dados relevantes que não estão contidos em seu treinamento original, melhorando assim a precisão e a relevância das respostas geradas.

## Contexto e Importância
Na aula, discutimos como, até o momento, as interações com modelos de LLM eram baseadas em prompts que dependiam exclusivamente do conhecimento prévio do modelo. No entanto, muitas aplicações práticas exigem informações específicas que podem não estar disponíveis no modelo. É aqui que o RAG se torna crucial. Ao integrar dados externos, como documentos, PDFs ou informações da web, podemos criar aplicações mais personalizadas e eficazes.

Por exemplo, ao desenvolver um chatbot para a Asimov, simplesmente usar um modelo de LLM não seria suficiente, pois ele não teria informações específicas sobre os cursos oferecidos. Com RAG, podemos fornecer ao modelo informações relevantes sobre a Asimov, permitindo que ele responda a perguntas específicas de forma precisa.

## Como Funciona o RAG?
O processo de RAG pode ser dividido em várias etapas:

### 1. Carregamento de Documentos (Document Loading):
    Nesta fase, os dados são carregados de diferentes fontes, como PDFs, CSVs ou bancos de dados. O LangChain oferece diversas ferramentas para facilitar esse carregamento.
### 2. Divisão de Documentos (Document Splitting):
    Os dados carregados são divididos em trechos menores. Essa divisão deve ser feita de forma inteligente para preservar o contexto e garantir que as informações relevantes sejam mantidas.
### 3. Embedding:
    Os trechos de texto são convertidos em vetores numéricos. Essa transformação é essencial para permitir comparações semânticas entre os textos, facilitando a busca por informações relevantes.
### 4. Armazenamento em VectorStore:
    Os vetores gerados são armazenados em uma base de dados específica para vetores, chamada VectorStore. Isso permite uma recuperação eficiente de informações.
### 5. Recuperação (Retrieval):
    Quando um usuário faz uma pergunta, o vetor correspondente à pergunta é gerado e comparado com os vetores armazenados na VectorStore. Os textos mais próximos são recuperados e utilizados para gerar uma resposta personalizada.

## Desafios do RAG
Um dos principais desafios do RAG é a limitação de tokens dos modelos de LLM. Por exemplo, o ChatGPT 3.5 aceita uma entrada de até 16 mil tokens. Portanto, é fundamental garantir que os trechos de texto que estamos passando para o modelo sejam significativos e não excedam esse limite.

## Pontos Importantes a Lembrar
- RAG combina geração de texto com recuperação de informações externas, permitindo respostas mais precisas e relevantes.
- As etapas do RAG incluem: Carregamento de documentos, divisão de documentos, embedding, armazenamento em VectorStore e recuperação.
- A qualidade das respostas depende da relevância dos dados externos que são fornecidos ao modelo.
- Os modelos de LLM têm limitações de tokens, o que exige cuidado na seleção dos trechos de texto a serem utilizados.
- RAG é uma técnica poderosa para criar aplicações personalizadas, especialmente em contextos onde informações específicas são necessárias.
- Compreender e aplicar a técnica de RAG pode transformar a forma como interagimos com modelos de linguagem, permitindo que criemos soluções mais eficazes e adaptadas às necessidades dos usuários.
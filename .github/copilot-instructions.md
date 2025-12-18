# Copilot Instructions for LangChain Project

## Project Overview
- This project is a Python-based framework for building, experimenting, and composing LLM-powered agents, chains, and workflows using [LangChain](https://python.langchain.com/).
- The codebase is organized for hands-on learning and prototyping with both OpenAI and HuggingFace models, prompt engineering, memory, and RAG (Retrieval-Augmented Generation) patterns.

## Key Directories & Files
- `main.py`: Example of a multi-step chain (translation + summarization) using structured outputs.
- `source/poc_langchain/`: Main source code. Contains submodules for environment config, OpenAI, HuggingFace, and RAG patterns.
  - `OpenAI/`: Chains, memory, routing, and prompt templates for OpenAI models.
  - `huggingface.py`: Example of using HuggingFace models via LangChain.
  - `environs.py`: Loads environment variables (via dotenv) for API keys/config.
- `aulas_curso_langchain/`: Jupyter notebooks for step-by-step tutorials and experiments.
- `base/requirements.txt` and `pyproject.toml`: All dependencies are managed here. Use Poetry or pip as preferred.

## Patterns & Conventions
- **Chains**: Compose multiple LLM calls using the `|` operator (e.g., translation → summarization).
- **Prompt Templates**: Use `ChatPromptTemplate` for both system and user prompts. Templates are often defined inline for clarity.
- **Structured Output**: Use Pydantic models to define expected output schemas for LLM responses.
- **Memory**: Use `InMemoryChatMessageHistory` and `RunnableWithMessageHistory` for session-based conversational memory. See `OpenAI/memory.py` for an example.
- **Routing**: Use Pydantic enums and models to categorize and route questions to specialized chains (see `OpenAI/roteamento.py`).
- **Parallel Chains**: Use `RunnableParallel` to run multiple chains concurrently (see `OpenAI/chains_paralelas.py`).
- **Environment**: Always load environment variables via `Environs` before instantiating models.

## Developer Workflows
- **Install dependencies**: `poetry install` or `pip install -r base/requirements.txt`
- **Run examples**: Execute any script directly (e.g., `python main.py` or `python source/poc_langchain/OpenAI/memory.py`).
- **Experiment**: Use the Jupyter notebooks in `aulas_curso_langchain/` for guided, interactive exploration.
- **Add new chains**: Place new chain scripts in `source/poc_langchain/OpenAI/` or as appropriate for the model/provider.

## External Integrations
- **OpenAI**: Requires API key in environment variables (see `.env` loading in `environs.py`).
- **HuggingFace**: Uses public models or requires HuggingFace Hub token in environment.

## Examples
- See `OpenAI/chain_1.py` and `main.py` for multi-step chains with structured outputs.
- See `OpenAI/memory.py` for session-based memory and chat history.
- See `OpenAI/roteamento.py` for dynamic routing based on question category.

## Tips for AI Agents
- Always check for environment variable loading before model instantiation.
- Follow the pattern of composing chains using the `|` operator.
- Use Pydantic models for output validation and schema enforcement.
- Reference the Jupyter notebooks for canonical usage patterns and experiments.

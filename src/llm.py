from langchain_ollama import OllamaLLM

def get_llm(model="llama3"):

    return OllamaLLM(model=model)
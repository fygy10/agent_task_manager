import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def setup_environment():
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")
    
    # Check for LangChain API key (optional)
    if os.environ.get('LANGCHAIN_API_KEY'):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_PROJECT"] = "langchain-academy"
        print("LangChain tracing enabled.")
    else:
        print("LangChain API key not found. Tracing will be disabled.")
    
    # Check for Tavily API key with detailed debugging
    tavily_api_key = os.environ.get('TAVILY_API_KEY')
    if not tavily_api_key:
        print("Warning: TAVILY_API_KEY not found. Search functionality will be disabled.")
    else:
        # Display first few characters of the key for verification
        masked_key = tavily_api_key[:5] + "..." + tavily_api_key[-3:] if len(tavily_api_key) > 8 else "***"
        print(f"Tavily API key found: {masked_key}. Length: {len(tavily_api_key)} characters")
        print("Search functionality enabled.")
    
    # Initialize and return the language model
    return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=1000)
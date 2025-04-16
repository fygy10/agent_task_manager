import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# First, try to import the Tavily SDK
try:
    from tavily import TavilyClient
    TAVILY_SDK_AVAILABLE = True
    print("Tavily SDK found. Using official SDK for search.")
except ImportError:
    TAVILY_SDK_AVAILABLE = False
    import requests
    print("Tavily SDK not found. Install with: pip install tavily-python")

def extract_location(text):
    """
    Extract location information from text.
    
    Args:
        text (str): User input text
        
    Returns:
        str: Extracted location or empty string
    """
    # Simple location extraction
    if "in" in text.lower():
        parts = text.lower().split("in")
        if len(parts) > 1:
            return parts[1].strip()
    
    return ""

def perform_search(query):
    """
    Perform a search using the Tavily API.
    
    Args:
        query (str): The search query
        
    Returns:
        dict: Search results or error message
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "Tavily API key not found in environment variables"}
    
    # Print API key information for debugging (not the full key)
    if len(api_key) > 7:
        print(f"Using Tavily API key: {api_key[:4]}...{api_key[-3:]}. Length: {len(api_key)} characters")
    else:
        print(f"API key may be malformed - length is only {len(api_key)} characters")
    
    # Check if API key has correct format (should start with tvly-)
    if not api_key.startswith("tvly-"):
        print("Warning: Tavily API key should start with 'tvly-'. Please check your API key format.")
    
    print(f"Search query: {query}")
    
    # If Tavily SDK is available, use that (recommended approach)
    if TAVILY_SDK_AVAILABLE:
        try:
            # Create client and perform search
            tavily_client = TavilyClient(api_key=api_key.strip())
            response = tavily_client.search(query, search_depth="basic", max_results=5)
            return response
        except Exception as e:
            print(f"Tavily SDK error: {str(e)}")
            return {"error": f"Search request failed: {str(e)}"}
    # Otherwise use direct REST API
    else:
        url = "https://api.tavily.com/search"
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key.strip()  # Ensure no whitespace
        }
        
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": 5
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            # Print status code for debugging
            print(f"Response status code: {response.status_code}")
            
            # If error response, print more details
            if response.status_code != 200:
                print(f"Error response: {response.text}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": f"Search request failed: {str(e)}"}

def perform_search_with_retry(query, max_retries=2):
    """
    Perform a search with retry logic for failed attempts.
    
    Args:
        query (str): The search query
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        dict: Search results or error message
    """
    for attempt in range(max_retries + 1):
        results = perform_search(query)
        
        # If successful or last attempt, return results
        if "error" not in results or attempt == max_retries:
            return results
        
        # Modify query slightly for retry
        query = f"information about {query}" if attempt == 0 else f"details on {query}"
        
        print(f"Retrying search with modified query: {query}")
    
    return results

def format_search_results(model, query, results, todo_list=None):
    """
    Format search results into a readable response with task context.
    
    Args:
        model: The language model
        query (str): Original search query
        results (dict): Search results from Tavily
        todo_list (list, optional): User's task list for context
        
    Returns:
        str: Formatted response
    """
    # Format search results for the model
    formatted_results = ""
    if "results" in results:
        for i, result in enumerate(results["results"], 1):
            formatted_results += f"\nResult {i}:\n"
            formatted_results += f"Title: {result.get('title', 'No title')}\n"
            formatted_results += f"Content: {result.get('content', 'No content')}\n"
            formatted_results += f"URL: {result.get('url', 'No URL')}\n"
    
    task_context = ""
    if todo_list:
        task_context = f"\nUser's tasks:\n{todo_list}"
    
    system_message = """
    You are a helpful assistant that creates useful responses based on search results.
    
    1. Summarize the most relevant information from the search results that addresses the user's query
    2. Format your response in a conversational and helpful way
    3. Include 1-2 specific details from the search results when relevant
    4. If the search results don't address the query, acknowledge this and suggest alternatives
    5. NEVER fabricate information not found in the search results
    6. Keep your response concise (1-2 paragraphs maximum)
    7. Always cite sources by including the source URL
    8. If the user has related tasks, connect the search results to those tasks when possible
    
    Remember to cite sources properly.
    """
    
    response = model.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"User Query: {query}\n{task_context}\n\nSearch Results: {formatted_results}")
    ])
    
    return response.content if hasattr(response, 'content') else str(response)

def should_search(model, text, todo_list=None):
    """
    Use the LLM to determine if text should trigger a search.
    """
    # Quick check for common task indicators that should NOT trigger search
    lower_text = text.lower()
    task_indicators = ["i need", "i want", "i should", "i must", "need to", 
                      "going to", "have to", "should", "must", "will"]
    
    # If text contains task indicators and doesn't have search-specific language,
    # treat it as a task, not a search
    if any(indicator in lower_text for indicator in task_indicators) and not any(
        keyword in lower_text for keyword in ["where", "how", "what is", "who is", 
                                              "when", "why", "find", "search", "look up"]):
        return False
    
    # More specific system message that understands task management context
    system_message = """
    You are analyzing messages in a task management application context.
    
    Determine if this user message requires searching the web for information, or if it's:
    1. Adding a new task to their todo list
    2. Asking about existing tasks
    3. Simple conversation with the assistant
    
    Messages like "I need a book" or "I want to buy shoes" are task additions, NOT search requests.
    Messages like "Where can I buy a book?" or "What are good running shoes?" ARE search requests.
    Messages like "Show me my tasks" or "What's on my list?" are task queries, NOT search requests.
    
    Respond with ONLY 'yes' if this requires a web search or 'no' if this is a task-related message.
    """
    
    # Add todo context if available
    context = ""
    if todo_list:
        context = f"User's current tasks:\n{todo_list}"
    
    response = model.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"{context}\n\nUser message: {text}")
    ])
    
    return "yes" in response.content.lower()

def handle_search_request(model, text, context=None, todo_list=None):
    """
    Handle a search request from user input with task context.
    
    Args:
        model: The language model
        text (str): User input text
        context (dict, optional): Additional context like user profile
        todo_list (list, optional): User's task list
        
    Returns:
        str: Search response or None if no search performed
    """
    # Use LLM to determine search intent
    if not should_search(model, text, todo_list):
        return None
    
    # Use LLM to extract search terms
    system_message = """
    Extract the key search terms from this user message.
    If the message relates to a task, identify the task and what information is needed.
    Format your response as a concise search query.
    """
    
    task_context = ""
    if todo_list:
        task_context = f"User's current tasks:\n{todo_list}"
    
    response = model.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"{task_context}\n\nUser message: {text}")
    ])
    
    search_query = response.content.strip()
    print(f"Search query: {search_query}")
    
    # Add location context if available
    location = extract_location(text)
    if not location and context and "location" in context:
        location = context["location"]
        if location and location not in search_query:
            search_query += f" in {location}"
    
    print(f"Searching for: {search_query}")
    
    # Perform search with retry
    search_results = perform_search_with_retry(search_query)
    
    if "error" in search_results:
        return f"Sorry, I couldn't complete the search: {search_results['error']}"
    
    # Format results with task context
    return format_search_results(model, search_query, search_results, todo_list)
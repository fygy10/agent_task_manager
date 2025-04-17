# import streamlit as st
# from datetime import datetime
# import uuid
# import os
# from dotenv import load_dotenv
# import time

# # Attempt to import OpenAI components
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# # Custom imports from your modules
# # You can adapt these imports to your actual file structure
# from models import MODEL_SYSTEM_MESSAGE, TRUSTCALL_INSTRUCTION
# from configuration import Configuration
# from profile import UpdateMemory, ToDo, Profile
# from trustcall import create_extractor

# # Initialize session state
# def init_session_state():
#     if 'store' not in st.session_state:
#         st.session_state.store = SimpleStore()
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'model' not in st.session_state:
#         st.session_state.model = None
#     if 'config' not in st.session_state:
#         st.session_state.config = Configuration(
#             user_id="user_" + str(uuid.uuid4())[:8],
#             todo_category="personal",
#             task_manager_role="You are an AI assistant that helps users organize their tasks and manage their to-do lists. You are friendly, helpful, and efficient."
#         )

# # SimpleStore implementation (in-memory)
# class SimpleStore:
#     def __init__(self):
#         self.data = {}
    
#     def search(self, namespace, key=None):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             return []
        
#         if key is not None:
#             if key in self.data[namespace_str]:
#                 return [type('obj', (object,), {'key': key, 'value': self.data[namespace_str][key]})]
#             return []
        
#         return [type('obj', (object,), {'key': k, 'value': v}) for k, v in self.data[namespace_str].items()]
    
#     def put(self, namespace, key, value):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             self.data[namespace_str] = {}
#         self.data[namespace_str][key] = value
#         return True

#     def remove(self, namespace, key):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             return False
        
#         if key not in self.data[namespace_str]:
#             return False
        
#         del self.data[namespace_str][key]
#         return True

# # Setup environment function
# def setup_environment():
#     # Load environment variables
#     load_dotenv()
    
#     # Check for OpenAI API key
#     if not os.environ.get('OPENAI_API_KEY'):
#         st.error("OPENAI_API_KEY not found. Please add it to your .env file or enter it below.")
#         api_key = st.text_input("Enter your OpenAI API Key:", type="password")
#         if api_key:
#             os.environ['OPENAI_API_KEY'] = api_key
#         else:
#             return None
    
#     # Initialize and return the language model
#     try:
#         return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=1000)
#     except Exception as e:
#         st.error(f"Error initializing language model: {str(e)}")
#         return None

# # Determine input type using LLM
# def determine_input_type(model_instance, text):
#     system_message = """
#     Analyze the user's input and classify it as one of the following types:
#     1. "question" - The user is asking for information or recommendations
#     2. "task" - The user is describing a task they want to complete or add to their list
#     3. "conversation" - The user is making a simple conversational response
    
#     Reply ONLY with one of these three words based on your classification.
#     """
    
#     response = model_instance.invoke([
#         SystemMessage(content=system_message),
#         HumanMessage(content=text)
#     ])
    
#     content = response.content.lower().strip() if hasattr(response, 'content') else ""
    
#     if "question" in content:
#         return "question"
#     elif "task" in content:
#         return "task"
#     else:
#         return "conversation"

# # Check if input might be a completion/removal message 
# def is_task_completion_message(text):
#     """Check if a message indicates a task has been completed."""
#     lower_text = text.lower()
#     completion_indicators = [
#         "completed", "finished", "done with", "did that", 
#         "bought", "purchased", "acquired", "got a", "went to", 
#         "visited", "checked out", "looked at", "saw the",
#         "you can remove", "delete", "take off", "cross off"
#     ]
#     return any(indicator in lower_text for indicator in completion_indicators)

# # Process user input
# def process_user_input(model_instance, store, user_input, config):
#     # Check if this is a task completion message before determining input type
#     if is_task_completion_message(user_input):
#         # Let the task removal handler deal with this
#         removal_response = handle_task_removal(user_input, store, config, model_instance)
#         if removal_response:
#             return {
#                 "type": "removal",
#                 "response": removal_response
#             }
    
#     # Create extractors
#     profile_extractor = create_extractor(
#         model=model_instance, 
#         tools=[Profile], 
#         tool_choice="Profile"
#     )
    
#     # Determine input type for normal processing
#     input_type = determine_input_type(model_instance, user_input)
    
#     # Create message
#     messages = [HumanMessage(content=user_input)]
    
#     # Get namespaces
#     namespace_profile = ('profile', config.todo_category, config.user_id)
#     namespace_todo = ('todo', config.todo_category, config.user_id)
    
#     # Extract profile information
#     existing_profiles = store.search(namespace_profile)
#     existing_profile_data = None
#     if existing_profiles:
#         existing_profile_data = [(item.key, "Profile", item.value) for item in existing_profiles]
    
#     profile_result = profile_extractor.invoke({
#         'messages': messages,
#         'existing': existing_profile_data
#     })
    
#     # Save profile information
#     for r, rmeta in zip(profile_result.get('responses', []), profile_result.get('response_metadata', [])):
#         if hasattr(r, 'model_dump'):
#             store.put(namespace_profile, rmeta.get('json_doc_id', str(uuid.uuid4())), r.model_dump(mode='json'))
#         else:
#             store.put(namespace_profile, rmeta.get('json_doc_id', str(uuid.uuid4())), r)
    
#     # If this looks like a task completion message but wasn't handled by task removal,
#     # don't add it as a new task
#     if is_task_completion_message(user_input):
#         return {
#             "type": "conversation",
#             "response": "I understand you've completed a task. To help me remove it from your list, please tell me which task(s) you've completed."
#         }
    
#     # If task, process it
#     if input_type == "task":
#         # Check for completion-related messages again to avoid adding them as tasks
#         if any(phrase in user_input.lower() for phrase in ["can you remove", "please remove", "completed", "finished", "done with"]):
#             return {
#                 "type": "conversation",
#                 "response": None  # Let the LLM generate a response
#             }
            
#         # Create a simple task directly instead of using extractor (more reliable)
#         task_title = user_input
#         if len(task_title) > 100:
#             task_title = task_title[:97] + "..."
        
#         task_data = {
#             "task": task_title,
#             "time_to_complete": 60,
#             "solutions": ["Complete this task"],
#             "status": "not started"
#         }
        
#         task_id = str(uuid.uuid4())
#         store.put(namespace_todo, task_id, task_data)
        
#         return {
#             "type": "task",
#             "tasks_added": [task_title],
#             "response": f"I've added the task \"{task_title}\" to your list."
#         }
    
#     return {
#         "type": input_type,
#         "response": None
#     }

# # Get LLM response
# def get_llm_response(model_instance, messages, store, config):
#     # Get profile
#     namespace = ('profile', config.todo_category, config.user_id)
#     profile_memories = store.search(namespace)
#     user_profile = profile_memories[0].value if profile_memories else None

#     # Get tasks
#     namespace = ('todo', config.todo_category, config.user_id)
#     task_memories = store.search(namespace)
#     todo_list = '\n'.join(f'• {mem.value.get("task", "Unknown task")}' for mem in task_memories)

#     # Get instructions
#     namespace = ('instructions', config.todo_category, config.user_id)
#     instruction_memories = store.search(namespace)
#     instructions = instruction_memories[0].value if instruction_memories else ''

#     # Format system message
#     system_msg = MODEL_SYSTEM_MESSAGE.format(
#         task_manager_role=config.task_manager_role, 
#         user_profile=user_profile, 
#         todo=todo_list, 
#         instructions=instructions
#     )

#     # Generate response
#     response = model_instance.invoke(
#         [SystemMessage(content=system_msg)] + messages
#     )

#     return response.content if hasattr(response, 'content') else str(response)

# # Handle task removal
# def handle_task_removal(text, store, config, model_instance):
#     lower_text = text.lower()
#     removal_indicators = [
#         "remove", "delete", "complete", "mark as done", "done", "finished",
#         "completed", "remove that task", "take off", "cross off", "clear"
#     ]
    
#     is_removal_request = any(indicator in lower_text for indicator in removal_indicators)
    
#     if not is_removal_request:
#         return None
    
#     namespace = ('todo', config.todo_category, config.user_id)
#     tasks = store.search(namespace)
    
#     if not tasks:
#         return "You don't have any tasks to remove."
    
#     system_message = """
#     You are analyzing a user request to remove or complete tasks from a todo list.
#     Given the user's message and their current task list, identify which task(s) should be removed.
    
#     Return ONLY the index numbers (starting from 1) of tasks to remove, separated by commas.
#     If all tasks should be removed, return "ALL".
#     If no specific task can be identified, return "NONE".
#     """
    
#     task_list = "\n".join([f"{i+1}. {task.value.get('task', 'Unknown task')}" 
#                          for i, task in enumerate(tasks)])
    
#     response = model_instance.invoke([
#         SystemMessage(content=system_message),
#         HumanMessage(content=f"User message: {text}\n\nCurrent tasks:\n{task_list}")
#     ])
    
#     result = response.content.strip() if hasattr(response, 'content') else str(response)
    
#     removed_tasks = []
#     if result == "ALL":
#         for task in tasks:
#             store.remove(namespace, task.key)
#             removed_tasks.append(task.value.get('task', 'Unknown task'))
#     elif result != "NONE":
#         try:
#             indices = [int(idx.strip()) - 1 for idx in result.split(',')]
#             for idx in indices:
#                 if 0 <= idx < len(tasks):
#                     task = tasks[idx]
#                     store.remove(namespace, task.key)
#                     removed_tasks.append(task.value.get('task', 'Unknown task'))
#         except ValueError:
#             return None
    
#     if removed_tasks:
#         if len(removed_tasks) == 1:
#             return f"I've removed the task: \"{removed_tasks[0]}\" from your list."
#         else:
#             task_list = "\n".join([f"• {task}" for task in removed_tasks])
#             return f"I've removed these tasks from your list:\n{task_list}"
    
#     return None

# # Handle search requests with real search functionality
# def handle_search_request(model_instance, user_input, search_context=None, todo_list=None):
#     """
#     Handle a search request from user input using real search functionality.
    
#     Args:
#         model_instance: The language model
#         user_input (str): User input text
#         search_context (dict, optional): Additional context like user profile
#         todo_list (str, optional): Current todo list
        
#     Returns:
#         str: Search response or None if no search performed
#     """
#     # Import necessary functions from search_integration
#     try:
#         from search_integration import perform_search, format_search_results
#     except ImportError:
#         # Simplified fallback if module not available
#         def perform_search(query):
#             try:
#                 import requests
#                 url = "https://api.tavily.com/search"
#                 headers = {
#                     "content-type": "application/json"
#                 }
#                 if os.environ.get("TAVILY_API_KEY"):
#                     headers["x-api-key"] = os.environ.get("TAVILY_API_KEY")
                
#                 payload = {
#                     "query": query,
#                     "search_depth": "basic",
#                     "max_results": 5
#                 }
                
#                 response = requests.post(url, headers=headers, json=payload)
#                 response.raise_for_status()
#                 return response.json()
#             except Exception as e:
#                 return {"error": f"Search request failed: {str(e)}"}
        
#         def format_search_results(model, query, results):
#             if "error" in results:
#                 return f"Sorry, I couldn't complete the search: {results['error']}"
            
#             formatted_results = ""
#             if "results" in results:
#                 for i, result in enumerate(results["results"], 1):
#                     formatted_results += f"\nResult {i}:\n"
#                     formatted_results += f"Title: {result.get('title', 'No title')}\n"
#                     formatted_results += f"Content: {result.get('content', 'No content')}\n"
#                     formatted_results += f"URL: {result.get('url', 'No URL')}\n"
            
#             system_message = """
#             You are a helpful assistant that creates useful responses based on search results.
            
#             1. Summarize the most relevant information from the search results that addresses the user's query
#             2. Format your response in a conversational and helpful way
#             3. Include 1-2 specific details from the search results when relevant
#             4. If the search results don't address the query, acknowledge this and suggest alternatives
#             5. NEVER fabricate information not found in the search results
#             6. Keep your response concise (1-2 paragraphs maximum)
#             7. Always cite sources by including the source URL
            
#             Remember to cite sources properly.
#             """
            
#             response = model.invoke([
#                 SystemMessage(content=system_message),
#                 HumanMessage(content=f"User Query: {query}\n\nSearch Results: {formatted_results}")
#             ])
            
#             return response.content if hasattr(response, 'content') else str(response)
    
#     # Keywords that might indicate search intent
#     search_keywords = ["find", "search", "where", "stores", "shops", "locations", 
#                        "near", "nearby", "around", "show me", "tell me about"]
    
#     # Check if any keywords are in the text
#     if any(keyword in user_input.lower() for keyword in search_keywords):
#         # Extract location from context if available
#         location = ""
#         if search_context and "location" in search_context:
#             location = search_context["location"]
        
#         # Build search query
#         search_query = user_input
#         if location and location not in search_query:
#             search_query += f" in {location}"
        
#         # Perform search
#         search_results = perform_search(search_query)
        
#         if "error" in search_results:
#             return f"Sorry, I couldn't complete the search: {search_results['error']}"
        
#         # Format results
#         return format_search_results(model_instance, search_query, search_results)
    
#     return None

# # Display chat messages
# def display_chat_messages():
#     for message in st.session_state.messages:
#         if message["role"] == "user":
#             with st.chat_message("user"):
#                 st.write(message["content"])
#         else:
#             with st.chat_message("assistant"):
#                 st.write(message["content"])

# # Main Streamlit app
# def main():
#     st.set_page_config(
#         page_title="Task Manager Assistant",
#         page_icon="💼",
#         layout="wide"
#     )
    
#     # Initialize session state
#     init_session_state()
    
#     # Page layout
#     st.title("💼 Task Manager Assistant")
    
#     # Sidebar
#     with st.sidebar:
#         st.title("Settings")
        
#         # User ID
#         user_id = st.text_input("User ID:", value=st.session_state.config.user_id)
#         if user_id != st.session_state.config.user_id:
#             st.session_state.config.user_id = user_id
#             st.experimental_rerun()
        
#         # Category
#         category = st.selectbox("Task Category:", 
#                                ["personal", "work", "shopping", "health", "education"],
#                                index=0)
#         if category != st.session_state.config.todo_category:
#             st.session_state.config.todo_category = category
#             st.experimental_rerun()
        
#         # Model initialization
#         if not OPENAI_AVAILABLE:
#             st.error("OpenAI package not installed. Please install langchain_openai.")
#         elif st.session_state.model is None:
#             if st.button("Initialize Language Model"):
#                 with st.spinner("Initializing language model..."):
#                     st.session_state.model = setup_environment()
#                     if st.session_state.model:
#                         st.success("Language model initialized successfully!")
#         else:
#             st.success("Language model is ready")
        
#         # View tasks with remove buttons
#         st.subheader("Your Tasks")
#         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#         tasks = st.session_state.store.search(namespace)
        
#         if tasks:
#             for i, task in enumerate(tasks):
#                 task_data = task.value
#                 task_key = task.key
                
#                 col1, col2 = st.columns([4, 1])
#                 with col1:
#                     st.write(f"{i+1}. {task_data.get('task', 'Unknown task')}")
#                 with col2:
#                     if st.button("Done", key=f"done_{task_key}"):
#                         # Remove the task
#                         st.session_state.store.remove(namespace, task_key)
#                         # Add a message
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": f"I've removed the task: \"{task_data.get('task', 'Unknown task')}\" from your list."
#                         })
#                         st.experimental_rerun()
                
#                 # Optional task details with small font
#                 with st.expander("Details", expanded=False):
#                     st.write(f"Est. time: {task_data.get('time_to_complete', '?')} mins")
#                     if task_data.get('status') != 'not started':
#                         st.write(f"Status: {task_data.get('status', 'Unknown')}")
#                     if task_data.get('solutions'):
#                         st.write("Solutions:")
#                         for solution in task_data.get('solutions', []):
#                             st.write(f"- {solution}")
#         else:
#             st.write("No tasks yet. Add some using the chat!")
        
#         # Add "Clear All Tasks" button
#         if tasks and st.button("Clear All Tasks"):
#             for task in tasks:
#                 st.session_state.store.remove(namespace, task.key)
            
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": "I've removed all tasks from your list."
#             })
#             st.experimental_rerun()
    
#     # Main content area - Chat interface
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Display chat messages
#         display_chat_messages()
        
#         # Chat input
#         if st.session_state.model is not None:
#             user_input = st.chat_input("Type your message here...")
#             if user_input:
#                 # Add user message to chat
#                 st.session_state.messages.append({"role": "user", "content": user_input})
                
#                 # Display updated chat
#                 with st.chat_message("user"):
#                     st.write(user_input)
                
#                 # Process input
#                 with st.spinner("Thinking..."):
#                     # Order of operations is important:
                    
#                     # 1. First check for task removal request (high priority)
#                     removal_response = handle_task_removal(user_input, st.session_state.store, 
#                                                          st.session_state.config, st.session_state.model)
#                     if removal_response:
#                         response = removal_response
                    
#                     # 2. Then check for task list query
#                     elif any(query in user_input.lower() for query in ["my tasks", "my to do", "todo list", 
#                                                                   "what are my", "what do i have", "show me my"]):
#                         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#                         tasks = st.session_state.store.search(namespace)
                        
#                         if tasks:
#                             task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')}" 
#                                                  for task in tasks])
#                             response = f"Here's your current task list:\n\n{task_list}"
#                         else:
#                             response = "You don't have any tasks yet. Would you like to add some?"
                    
#                     # 3. Check for search request
#                     elif search_response := handle_search_request(st.session_state.model, user_input):
#                         response = search_response
                    
#                     # 4. Process with user input handler
#                     else:
#                         # Process with our enhanced input handler
#                         result = process_user_input(st.session_state.model, st.session_state.store, 
#                                                    user_input, st.session_state.config)
                        
#                         # If response from handler, use it
#                         if result and result.get("response"):
#                             response = result["response"]
#                         else:
#                             # Otherwise get LLM response
#                             messages = [{"role": "user" if msg["role"] == "user" else "assistant", "content": msg["content"]} 
#                                       for msg in st.session_state.messages]
#                             llm_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" 
#                                            else AIMessage(content=msg["content"]) 
#                                            for msg in st.session_state.messages]
#                             response = get_llm_response(st.session_state.model, llm_messages, 
#                                                       st.session_state.store, st.session_state.config)
                
#                 # Display assistant response
#                 with st.chat_message("assistant"):
#                     st.write(response)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response})
#         else:
#             st.info("Please initialize the language model in the sidebar to start chatting.")
    
#     with col2:
#         # Task summary
#         st.subheader("Task Summary")
#         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#         tasks = st.session_state.store.search(namespace)
        
#         total_tasks = len(tasks)
#         completed_tasks = sum(1 for task in tasks if task.value.get('status') in ['done', 'completed'])
#         in_progress_tasks = sum(1 for task in tasks if task.value.get('status') == 'in progress')
        
#         st.metric("Total Tasks", total_tasks)
#         st.metric("Completed", completed_tasks)
#         st.metric("In Progress", in_progress_tasks)
        
#         # Estimated time
#         total_time = sum(task.value.get('time_to_complete', 0) for task in tasks)
#         st.metric("Estimated Time (mins)", total_time)
        
#         # User profile with cleaner display
#         st.subheader("Your Profile")
#         namespace = ('profile', st.session_state.config.todo_category, st.session_state.config.user_id)
#         profiles = st.session_state.store.search(namespace)
        
#         profile_container = st.container(height=200)
#         with profile_container:
#             if profiles:
#                 profile = profiles[0].value
#                 if isinstance(profile, dict):
#                     profile_data = []
#                     if profile.get('name'):
#                         profile_data.append(f"**Name:** {profile.get('name')}")
#                     if profile.get('location'):
#                         profile_data.append(f"**Location:** {profile.get('location')}")
#                     if profile.get('job'):
#                         profile_data.append(f"**Job:** {profile.get('job')}")
#                     if profile.get('interests') and len(profile.get('interests')) > 0:
#                         profile_data.append(f"**Interests:** {', '.join(profile.get('interests'))}")
                    
#                     if profile_data:
#                         for item in profile_data:
#                             st.write(item)
#                     else:
#                         st.write("Limited profile information. Tell me more about yourself!")
#             else:
#                 st.write("No profile information yet. Tell me about yourself!")
        
#         # Quick actions with better UI
#         st.subheader("Quick Actions")
        
#         # Organized by category
#         task_categories = {
#             "Personal": ["Buy groceries", "Call mom", "Schedule doctor appointment"],
#             "Work": ["Send email", "Write report", "Schedule meeting"],
#             "Shopping": ["Buy groceries", "Buy birthday gift", "Order online"]
#         }
        
#         selected_category = st.selectbox("Category:", list(task_categories.keys()))
#         quick_tasks = task_categories[selected_category]
        
#         # Display quick task buttons in a grid
#         cols = st.columns(2)
#         for i, task in enumerate(quick_tasks):
#             with cols[i % 2]:
#                 if st.button(f"Add: {task}", key=f"quick_{selected_category}_{task}"):
#                     task_data = {
#                         "task": task,
#                         "time_to_complete": 30,
#                         "solutions": ["Complete this task"],
#                         "status": "not started"
#                     }
                    
#                     namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#                     task_id = str(uuid.uuid4())
#                     st.session_state.store.put(namespace, task_id, task_data)
                    
#                     # Add system message
#                     st.session_state.messages.append({"role": "assistant", "content": f"I've added the task \"{task}\" to your list."})
#                     st.experimental_rerun()

# if __name__ == "__main__":
#     main()


# import streamlit as st
# from datetime import datetime
# import uuid
# import os
# from dotenv import load_dotenv
# import time

# # Attempt to import OpenAI components
# try:
#     from langchain_openai import ChatOpenAI
#     from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False

# # Custom imports from your modules
# from models import MODEL_SYSTEM_MESSAGE, TRUSTCALL_INSTRUCTION
# from configuration import Configuration
# from profile import UpdateMemory, ToDo, Profile
# from trustcall import create_extractor
# from graph import run_graph  # Import the run_graph function

# # Import search integration if available
# try:
#     from search_integration import handle_search_request, should_search
#     SEARCH_AVAILABLE = True
# except ImportError:
#     SEARCH_AVAILABLE = False

# # Initialize session state
# def init_session_state():
#     if 'store' not in st.session_state:
#         st.session_state.store = SimpleStore()
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'model' not in st.session_state:
#         st.session_state.model = None
#     if 'config' not in st.session_state:
#         st.session_state.config = Configuration(
#             user_id="user_" + str(uuid.uuid4())[:8],
#             todo_category="personal",
#             task_manager_role="You are an AI assistant that helps users organize their tasks and manage their to-do lists. You are friendly, helpful, and efficient."
#         )

# # SimpleStore implementation (in-memory)
# class SimpleStore:
#     def __init__(self):
#         self.data = {}
    
#     def search(self, namespace, key=None):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             return []
        
#         if key is not None:
#             if key in self.data[namespace_str]:
#                 return [type('obj', (object,), {'key': key, 'value': self.data[namespace_str][key]})]
#             return []
        
#         return [type('obj', (object,), {'key': k, 'value': v}) for k, v in self.data[namespace_str].items()]
    
#     def put(self, namespace, key, value):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             self.data[namespace_str] = {}
#         self.data[namespace_str][key] = value
#         return True

#     def remove(self, namespace, key):
#         namespace_str = "_".join(str(part) for part in namespace)
#         if namespace_str not in self.data:
#             return False
        
#         if key not in self.data[namespace_str]:
#             return False
        
#         del self.data[namespace_str][key]
#         return True

# # Setup environment function
# def setup_environment():
#     # Load environment variables
#     load_dotenv()
    
#     # Check for OpenAI API key
#     if not os.environ.get('OPENAI_API_KEY'):
#         st.error("OPENAI_API_KEY not found. Please add it to your .env file or enter it below.")
#         api_key = st.text_input("Enter your OpenAI API Key:", type="password")
#         if api_key:
#             os.environ['OPENAI_API_KEY'] = api_key
#         else:
#             return None
    
#     # Initialize and return the language model
#     try:
#         return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=1000)
#     except Exception as e:
#         st.error(f"Error initializing language model: {str(e)}")
#         return None

# # Process user input directly without relying on child_runs
# def process_user_input(model_instance, store, user_input, config):
#     """Process user input and determine appropriate actions."""
    
#     # Define namespaces
#     namespace_profile = ('profile', config.todo_category, config.user_id)
#     namespace_todo = ('todo', config.todo_category, config.user_id)
    
#     # Check for task indicators
#     task_indicators = ["i need", "i want", "i should", "i must", "need to", 
#                       "going to", "have to", "should", "must", "will"]
                      
#     # If text contains task indicators and doesn't look like a question or search
#     if any(indicator in user_input.lower() for indicator in task_indicators) and not any(
#         keyword in user_input.lower() for keyword in ["where", "how", "what is", "who is", 
#                                               "when", "why", "find", "search", "look up"]) and "?" not in user_input:
        
#         # Create a simple task directly
#         task_data = {
#             "task": user_input,
#             "time_to_complete": 60,
#             "solutions": ["Complete this task"],
#             "status": "not started"
#         }
        
#         # Add to store
#         task_id = str(uuid.uuid4())
#         store.put(namespace_todo, task_id, task_data)
        
#         return f"I've added the task \"{user_input}\" to your list."
    
#     # If it's not clearly a task, use the graph processing
#     return None

# def handle_task_removal(text, store, config, model_instance):
#     """Check if a message indicates task completion and remove completed tasks."""
#     lower_text = text.lower()
#     removal_indicators = [
#         "remove", "delete", "complete", "mark as done", "done", "finished",
#         "completed", "remove that task", "take off", "cross off", "clear"
#     ]
    
#     is_removal_request = any(indicator in lower_text for indicator in removal_indicators)
    
#     if not is_removal_request:
#         return None
    
#     namespace = ('todo', config.todo_category, config.user_id)
#     tasks = store.search(namespace)
    
#     if not tasks:
#         return "You don't have any tasks to remove."
    
#     # For simple case - just remove the most recent task related to a book if "book" is mentioned
#     if "book" in lower_text and any(task.value.get('task', '').lower().find('book') != -1 for task in tasks):
#         for task in tasks:
#             if 'book' in task.value.get('task', '').lower():
#                 store.remove(namespace, task.key)
#                 return f"I've removed the task: \"{task.value.get('task', 'Unknown task')}\" from your list."
    
#     # More complex case - use LLM to identify tasks to remove
#     system_message = """
#     You are analyzing a user request to remove or complete tasks from a todo list.
#     Given the user's message and their current task list, identify which task(s) should be removed.
    
#     Return ONLY the index numbers (starting from 1) of tasks to remove, separated by commas.
#     If all tasks should be removed, return "ALL".
#     If no specific task can be identified, return "NONE".
#     """
    
#     task_list = "\n".join([f"{i+1}. {task.value.get('task', 'Unknown task')}" 
#                          for i, task in enumerate(tasks)])
    
#     response = model_instance.invoke([
#         SystemMessage(content=system_message),
#         HumanMessage(content=f"User message: {text}\n\nCurrent tasks:\n{task_list}")
#     ])
    
#     result = response.content.strip() if hasattr(response, 'content') else str(response)
    
#     removed_tasks = []
#     if result == "ALL":
#         for task in list(tasks):  # Create a copy of the list to safely iterate and modify
#             store.remove(namespace, task.key)
#             removed_tasks.append(task.value.get('task', 'Unknown task'))
#     elif result != "NONE":
#         try:
#             indices = [int(idx.strip()) - 1 for idx in result.split(',')]
#             # Sort indices in descending order to avoid index shifting when removing items
#             for idx in sorted(indices, reverse=True):
#                 if 0 <= idx < len(tasks):
#                     task = tasks[idx]
#                     store.remove(namespace, task.key)
#                     removed_tasks.append(task.value.get('task', 'Unknown task'))
#         except ValueError:
#             pass
    
#     # Verify removal success - check if tasks were actually removed
#     updated_tasks = store.search(namespace)
#     if len(updated_tasks) == len(tasks) and removed_tasks:
#         # Tasks weren't actually removed - try direct removal approach
#         for task in list(tasks):
#             if any(removed_task.lower() in task.value.get('task', '').lower() for removed_task in removed_tasks):
#                 store.remove(namespace, task.key)
    
#     if removed_tasks:
#         if len(removed_tasks) == 1:
#             return f"I've removed the task: \"{removed_tasks[0]}\" from your list."
#         else:
#             task_list = "\n".join([f"• {task}" for task in removed_tasks])
#             return f"I've removed these tasks from your list:\n{task_list}"
    
#     return None

# # Get LLM response without using the graph
# def get_llm_response(model_instance, messages, store, config):
#     """Generate a response using the LLM with context."""
#     # Get profile
#     namespace = ('profile', config.todo_category, config.user_id)
#     profile_memories = store.search(namespace)
#     user_profile = profile_memories[0].value if profile_memories else None

#     # Get tasks
#     namespace = ('todo', config.todo_category, config.user_id)
#     task_memories = store.search(namespace)
#     todo_list = '\n'.join(f'• {mem.value.get("task", "Unknown task")}' for mem in task_memories)

#     # Get instructions
#     namespace = ('instructions', config.todo_category, config.user_id)
#     instruction_memories = store.search(namespace)
#     instructions = instruction_memories[0].value if instruction_memories else ''

#     # Format system message
#     system_msg = MODEL_SYSTEM_MESSAGE.format(
#         task_manager_role=config.task_manager_role, 
#         user_profile=user_profile, 
#         todo=todo_list, 
#         instructions=instructions
#     )

#     # Generate response
#     response = model_instance.invoke(
#         [SystemMessage(content=system_msg)] + messages
#     )

#     return response.content if hasattr(response, 'content') else str(response)

# # Main Streamlit app
# def main():
#     st.set_page_config(
#         page_title="Task Manager Assistant",
#         page_icon="💼",
#         layout="wide"
#     )
    
#     # Initialize session state
#     init_session_state()
    
#     # Page layout
#     st.title("💼 Task Manager Assistant")
    
#     # Sidebar
#     with st.sidebar:
#         st.title("Settings")
        
#         # User ID
#         user_id = st.text_input("User ID:", value=st.session_state.config.user_id)
#         if user_id != st.session_state.config.user_id:
#             st.session_state.config.user_id = user_id
#             st.experimental_rerun()
        
#         # Category
#         category = st.selectbox("Task Category:", 
#                                ["personal", "work", "shopping", "health", "education"],
#                                index=0)
#         if category != st.session_state.config.todo_category:
#             st.session_state.config.todo_category = category
#             st.experimental_rerun()
        
#         # Model initialization
#         if not OPENAI_AVAILABLE:
#             st.error("OpenAI package not installed. Please install langchain_openai.")
#         elif st.session_state.model is None:
#             if st.button("Initialize Language Model"):
#                 with st.spinner("Initializing language model..."):
#                     st.session_state.model = setup_environment()
#                     if st.session_state.model:
#                         st.success("Language model initialized successfully!")
#         else:
#             st.success("Language model is ready")
        
#         # View tasks with remove buttons
#         st.subheader("Your Tasks")
#         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#         tasks = st.session_state.store.search(namespace)
        
#         if tasks:
#             for i, task in enumerate(tasks):
#                 task_data = task.value
#                 task_key = task.key
                
#                 col1, col2 = st.columns([4, 1])
#                 with col1:
#                     st.write(f"{i+1}. {task_data.get('task', 'Unknown task')}")
#                 with col2:
#                     if st.button("Done", key=f"done_{task_key}"):
#                         # Remove the task
#                         st.session_state.store.remove(namespace, task_key)
#                         # Add a message
#                         st.session_state.messages.append({
#                             "role": "assistant", 
#                             "content": f"I've removed the task: \"{task_data.get('task', 'Unknown task')}\" from your list."
#                         })
#                         st.experimental_rerun()
                
#                 # Optional task details with small font
#                 with st.expander("Details", expanded=False):
#                     st.write(f"Est. time: {task_data.get('time_to_complete', '?')} mins")
#                     if task_data.get('status') != 'not started':
#                         st.write(f"Status: {task_data.get('status', 'Unknown')}")
#                     if task_data.get('solutions'):
#                         st.write("Solutions:")
#                         for solution in task_data.get('solutions', []):
#                             st.write(f"- {solution}")
#         else:
#             st.write("No tasks yet. Add some using the chat!")
        
#         # Add "Clear All Tasks" button
#         if tasks and st.button("Clear All Tasks"):
#             for task in tasks:
#                 st.session_state.store.remove(namespace, task.key)
            
#             st.session_state.messages.append({
#                 "role": "assistant", 
#                 "content": "I've removed all tasks from your list."
#             })
#             st.experimental_rerun()
    
#     # Main content area - Chat interface
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         # Display chat messages
#         for message in st.session_state.messages:
#             if message["role"] == "user":
#                 with st.chat_message("user"):
#                     st.write(message["content"])
#             else:
#                 with st.chat_message("assistant"):
#                     st.write(message["content"])
        
#         # Chat input
#         if st.session_state.model is not None:
#             user_input = st.chat_input("Type your message here...")
#             if user_input:
#                 # Add user message to chat
#                 st.session_state.messages.append({"role": "user", "content": user_input})
                
#                 # Display updated chat
#                 with st.chat_message("user"):
#                     st.write(user_input)
                
#                 # Process input
#                 with st.spinner("Thinking..."):
#                     # Order of operations is important:
                    
#                     # 1. First check for task removal request (high priority)
#                     removal_response = handle_task_removal(user_input, st.session_state.store, 
#                                                          st.session_state.config, st.session_state.model)
#                     if removal_response:
#                         response = removal_response
                    
#                     # 2. Then check for task list query
#                     elif any(query in user_input.lower() for query in ["my tasks", "my to do", "todo list", 
#                                                                   "what are my", "what do i have", "show me my"]):
#                         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#                         tasks = st.session_state.store.search(namespace)
                        
#                         if tasks:
#                             task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')}" 
#                                                  for task in tasks])
#                             response = f"Here's your current task list:\n\n{task_list}"
#                         else:
#                             response = "You don't have any tasks yet. Would you like to add some?"
                    
#                     # 3. Process direct task addition without needing graph
#                     elif direct_response := process_user_input(st.session_state.model, st.session_state.store, 
#                                                              user_input, st.session_state.config):
#                         response = direct_response
                    
#                     # 4. Check for search request (if available)
#                     elif SEARCH_AVAILABLE and should_search(st.session_state.model, user_input):
#                         # Get user profile for context
#                         namespace = ('profile', st.session_state.config.todo_category, st.session_state.config.user_id)
#                         profiles = st.session_state.store.search(namespace)
#                         user_profile = None
#                         if profiles:
#                             user_profile = profiles[0].value
                        
#                         # Get tasks for context
#                         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#                         tasks = st.session_state.store.search(namespace)
#                         todo_list = '\n'.join(f'• {task.value.get("task", "Unknown task")}' for task in tasks)
                        
#                         search_response = handle_search_request(st.session_state.model, user_input, user_profile, todo_list)
#                         if search_response:
#                             response = search_response
#                         else:
#                             # Fallback to direct LLM response
#                             llm_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" 
#                                           else AIMessage(content=msg["content"]) 
#                                           for msg in st.session_state.messages[:-1]]  # Exclude the current message
#                             llm_messages.append(HumanMessage(content=user_input))
#                             response = get_llm_response(st.session_state.model, llm_messages, 
#                                                      st.session_state.store, st.session_state.config)
                    
#                     # 5. Fallback to direct LLM response without using graph
#                     else:
#                         llm_messages = [HumanMessage(content=msg["content"]) if msg["role"] == "user" 
#                                       else AIMessage(content=msg["content"]) 
#                                       for msg in st.session_state.messages[:-1]]  # Exclude the current message
#                         llm_messages.append(HumanMessage(content=user_input))
#                         response = get_llm_response(st.session_state.model, llm_messages, 
#                                                  st.session_state.store, st.session_state.config)
                    
#                     # Display assistant response
#                     with st.chat_message("assistant"):
#                         st.write(response)
                    
#                     # Add assistant response to chat history
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#         else:
#             st.info("Please initialize the language model in the sidebar to start chatting.")
    
#     with col2:
#         # Task summary
#         st.subheader("Task Summary")
#         namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#         tasks = st.session_state.store.search(namespace)
        
#         total_tasks = len(tasks)
#         completed_tasks = sum(1 for task in tasks if task.value.get('status') in ['done', 'completed'])
#         in_progress_tasks = sum(1 for task in tasks if task.value.get('status') == 'in progress')
        
#         st.metric("Total Tasks", total_tasks)
#         st.metric("Completed", completed_tasks)
#         st.metric("In Progress", in_progress_tasks)
        
#         # Estimated time
#         total_time = sum(task.value.get('time_to_complete', 0) for task in tasks)
#         st.metric("Estimated Time (mins)", total_time)
        
#         # User profile with cleaner display
#         st.subheader("Your Profile")
#         namespace = ('profile', st.session_state.config.todo_category, st.session_state.config.user_id)
#         profiles = st.session_state.store.search(namespace)
        
#         profile_container = st.container(height=200)
#         with profile_container:
#             if profiles:
#                 profile = profiles[0].value
#                 if isinstance(profile, dict):
#                     profile_data = []
#                     if profile.get('name'):
#                         profile_data.append(f"**Name:** {profile.get('name')}")
#                     if profile.get('location'):
#                         profile_data.append(f"**Location:** {profile.get('location')}")
#                     if profile.get('job'):
#                         profile_data.append(f"**Job:** {profile.get('job')}")
#                     if profile.get('interests') and len(profile.get('interests')) > 0:
#                         profile_data.append(f"**Interests:** {', '.join(profile.get('interests'))}")
                    
#                     if profile_data:
#                         for item in profile_data:
#                             st.write(item)
#                     else:
#                         st.write("Limited profile information. Tell me more about yourself!")
#             else:
#                 st.write("No profile information yet. Tell me about yourself!")
        
#         # Quick actions with better UI
#         st.subheader("Quick Actions")
        
#         # Organized by category
#         task_categories = {
#             "Personal": ["Buy groceries", "Call mom", "Schedule doctor appointment"],
#             "Work": ["Send email", "Write report", "Schedule meeting"],
#             "Shopping": ["Buy groceries", "Buy birthday gift", "Order online"]
#         }
        
#         selected_category = st.selectbox("Category:", list(task_categories.keys()))
#         quick_tasks = task_categories[selected_category]
        
#         # Display quick task buttons in a grid
#         cols = st.columns(2)
#         for i, task in enumerate(quick_tasks):
#             with cols[i % 2]:
#                 if st.button(f"Add: {task}", key=f"quick_{selected_category}_{task}"):
#                     task_data = {
#                         "task": task,
#                         "time_to_complete": 30,
#                         "solutions": ["Complete this task"],
#                         "status": "not started"
#                     }
                    
#                     namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
#                     task_id = str(uuid.uuid4())
#                     st.session_state.store.put(namespace, task_id, task_data)
                    
#                     # Add system message
#                     st.session_state.messages.append({"role": "assistant", "content": f"I've added the task \"{task}\" to your list."})
#                     st.experimental_rerun()

# if __name__ == "__main__":
#     main()


import streamlit as st
from datetime import datetime
import uuid
import os
from dotenv import load_dotenv

# Attempt to import OpenAI components
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Custom imports from your modules
from models import MODEL_SYSTEM_MESSAGE
from configuration import Configuration
from profile import UpdateMemory
from graph import run_graph  # Import the run_graph function

# Initialize session state
def init_session_state():
    if 'store' not in st.session_state:
        st.session_state.store = SimpleStore()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'config' not in st.session_state:
        st.session_state.config = Configuration(
            user_id="user_" + str(uuid.uuid4())[:8],
            todo_category="personal",
            task_manager_role="You are an AI assistant that helps users organize their tasks and manage their to-do lists. You are friendly, helpful, and efficient."
        )

# SimpleStore implementation (in-memory)
class SimpleStore:
    def __init__(self):
        self.data = {}
    
    def search(self, namespace, key=None):
        namespace_str = "_".join(str(part) for part in namespace)
        if namespace_str not in self.data:
            return []
        
        if key is not None:
            if key in self.data[namespace_str]:
                return [type('obj', (object,), {'key': key, 'value': self.data[namespace_str][key]})]
            return []
        
        return [type('obj', (object,), {'key': k, 'value': v}) for k, v in self.data[namespace_str].items()]
    
    def put(self, namespace, key, value):
        namespace_str = "_".join(str(part) for part in namespace)
        if namespace_str not in self.data:
            self.data[namespace_str] = {}
        self.data[namespace_str][key] = value
        return True

    def remove(self, namespace, key):
        namespace_str = "_".join(str(part) for part in namespace)
        if namespace_str not in self.data:
            return False
        
        if key not in self.data[namespace_str]:
            return False
        
        del self.data[namespace_str][key]
        return True

# Setup environment function
def setup_environment():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.environ.get('OPENAI_API_KEY'):
        st.error("OPENAI_API_KEY not found. Please add it to your .env file or enter it below.")
        api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        else:
            return None
    
    # Initialize and return the language model
    try:
        return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=1000)
    except Exception as e:
        st.error(f"Error initializing language model: {str(e)}")
        return None

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Task Manager Assistant",
        page_icon="💼",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Page layout
    st.title("💼 Task Manager Assistant")
    
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        
        # User ID
        user_id = st.text_input("User ID:", value=st.session_state.config.user_id)
        if user_id != st.session_state.config.user_id:
            st.session_state.config.user_id = user_id
            st.experimental_rerun()
        
        # Category
        category = st.selectbox("Task Category:", 
                               ["personal", "work", "shopping", "health", "education"],
                               index=0)
        if category != st.session_state.config.todo_category:
            st.session_state.config.todo_category = category
            st.experimental_rerun()
        
        # Model initialization
        if not OPENAI_AVAILABLE:
            st.error("OpenAI package not installed. Please install langchain_openai.")
        elif st.session_state.model is None:
            if st.button("Initialize Language Model"):
                with st.spinner("Initializing language model..."):
                    st.session_state.model = setup_environment()
                    if st.session_state.model:
                        st.success("Language model initialized successfully!")
        else:
            st.success("Language model is ready")
        
        # View tasks with remove buttons
        st.subheader("Your Tasks")
        namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
        tasks = st.session_state.store.search(namespace)
        
        if tasks:
            for i, task in enumerate(tasks):
                task_data = task.value
                task_key = task.key
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {task_data.get('task', 'Unknown task')}")
                with col2:
                    if st.button("Done", key=f"done_{task_key}"):
                        # Use graph to handle the task removal instead of direct store manipulation
                        llm_messages = list(st.session_state.messages)
                        # Add a special completion message for this task
                        completion_msg = HumanMessage(content=f"I've completed the task: {task_data.get('task', 'Unknown task')}")
                        
                        # Run graph with the completion message
                        result = run_graph(
                            messages=llm_messages + [completion_msg],
                            user_id=st.session_state.config.user_id,
                            todo_category=st.session_state.config.todo_category,
                            task_manager_role=st.session_state.config.task_manager_role,
                            store=st.session_state.store
                        )
                        
                        # Extract response
                        if result and "messages" in result and result["messages"]:
                            response_msg = result["messages"][-1]
                            response_content = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
                        else:
                            response_content = f"I've removed the task: \"{task_data.get('task', 'Unknown task')}\" from your list."
                        
                        # Add completion message and response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                        st.experimental_rerun()
                
                # Optional task details with small font
                with st.expander("Details", expanded=False):
                    st.write(f"Est. time: {task_data.get('time_to_complete', '?')} mins")
                    if task_data.get('status') != 'not started':
                        st.write(f"Status: {task_data.get('status', 'Unknown')}")
                    if task_data.get('solutions'):
                        st.write("Solutions:")
                        for solution in task_data.get('solutions', []):
                            st.write(f"- {solution}")
        else:
            st.write("No tasks yet. Add some using the chat!")
        
        # Add "Clear All Tasks" button
        if tasks and st.button("Clear All Tasks"):
            # Use graph to handle the task removal instead of direct store manipulation
            llm_messages = list(st.session_state.messages)
            # Add a special completion message for all tasks
            completion_msg = HumanMessage(content="I've completed all my tasks. Please clear my task list.")
            
            # Run graph with the completion message
            result = run_graph(
                messages=llm_messages + [completion_msg],
                user_id=st.session_state.config.user_id,
                todo_category=st.session_state.config.todo_category,
                task_manager_role=st.session_state.config.task_manager_role,
                store=st.session_state.store
            )
            
            # Extract response
            if result and "messages" in result and result["messages"]:
                response_msg = result["messages"][-1]
                response_content = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
            else:
                response_content = "I've removed all tasks from your list."
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_content})
            st.experimental_rerun()
    
    # Main content area - Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Chat input
        if st.session_state.model is not None:
            user_input = st.chat_input("Type your message here...")
            if user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Display updated chat
                with st.chat_message("user"):
                    st.write(user_input)
                
                # Process input using the LangGraph
                with st.spinner("Thinking..."):
                    # Convert dict messages to LangChain message objects
                    llm_messages = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            llm_messages.append(HumanMessage(content=msg["content"]))
                        else:
                            llm_messages.append(AIMessage(content=msg["content"]))
                    
                    # Ensure the last message is the user input
                    if not llm_messages or llm_messages[-1].content != user_input:
                        llm_messages.append(HumanMessage(content=user_input))
                    
                    try:
                        # Process through LangGraph
                        result = run_graph(
                            messages=llm_messages,
                            user_id=st.session_state.config.user_id,
                            todo_category=st.session_state.config.todo_category,
                            task_manager_role=st.session_state.config.task_manager_role,
                            store=st.session_state.store
                        )
                        
                        # Extract response from graph result
                        if result and "messages" in result and result["messages"]:
                            response_msg = result["messages"][-1]
                            response = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
                        else:
                            response = "I encountered an issue processing your request."
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        response = "I'm having trouble responding right now. Please try again."
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.info("Please initialize the language model in the sidebar to start chatting.")
    
    with col2:
        # Task summary
        st.subheader("Task Summary")
        namespace = ('todo', st.session_state.config.todo_category, st.session_state.config.user_id)
        tasks = st.session_state.store.search(namespace)
        
        total_tasks = len(tasks)
        completed_tasks = sum(1 for task in tasks if task.value.get('status') in ['done', 'completed'])
        in_progress_tasks = sum(1 for task in tasks if task.value.get('status') == 'in progress')
        
        st.metric("Total Tasks", total_tasks)
        st.metric("Completed", completed_tasks)
        st.metric("In Progress", in_progress_tasks)
        
        # Estimated time
        total_time = sum(task.value.get('time_to_complete', 0) for task in tasks)
        st.metric("Estimated Time (mins)", total_time)
        
        # User profile with cleaner display
        st.subheader("Your Profile")
        namespace = ('profile', st.session_state.config.todo_category, st.session_state.config.user_id)
        profiles = st.session_state.store.search(namespace)
        
        profile_container = st.container(height=200)
        with profile_container:
            if profiles:
                profile = profiles[0].value
                if isinstance(profile, dict):
                    profile_data = []
                    if profile.get('name'):
                        profile_data.append(f"**Name:** {profile.get('name')}")
                    if profile.get('location'):
                        profile_data.append(f"**Location:** {profile.get('location')}")
                    if profile.get('job'):
                        profile_data.append(f"**Job:** {profile.get('job')}")
                    if profile.get('interests') and len(profile.get('interests')) > 0:
                        profile_data.append(f"**Interests:** {', '.join(profile.get('interests'))}")
                    
                    if profile_data:
                        for item in profile_data:
                            st.write(item)
                    else:
                        st.write("Limited profile information. Tell me more about yourself!")
            else:
                st.write("No profile information yet. Tell me about yourself!")
        
        # Quick actions with better UI
        st.subheader("Quick Actions")
        
        # Organized by category
        task_categories = {
            "Personal": ["Buy groceries", "Call mom", "Schedule doctor appointment"],
            "Work": ["Send email", "Write report", "Schedule meeting"],
            "Shopping": ["Buy groceries", "Buy birthday gift", "Order online"]
        }
        
        selected_category = st.selectbox("Category:", list(task_categories.keys()))
        quick_tasks = task_categories[selected_category]
        
        # Display quick task buttons in a grid
        cols = st.columns(2)
        for i, task in enumerate(quick_tasks):
            with cols[i % 2]:
                if st.button(f"Add: {task}", key=f"quick_{selected_category}_{task}"):
                    # Use graph to add task instead of direct store manipulation
                    llm_messages = list(st.session_state.messages)
                    # Add a special task addition message
                    task_msg = HumanMessage(content=f"I need to {task}")
                    
                    # Run graph with the task message
                    result = run_graph(
                        messages=llm_messages + [task_msg],
                        user_id=st.session_state.config.user_id,
                        todo_category=st.session_state.config.todo_category,
                        task_manager_role=st.session_state.config.task_manager_role,
                        store=st.session_state.store
                    )
                    
                    # Extract response
                    if result and "messages" in result and result["messages"]:
                        response_msg = result["messages"][-1]
                        response_content = response_msg.content if hasattr(response_msg, "content") else str(response_msg)
                    else:
                        response_content = f"I've added the task \"{task}\" to your list."
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langchain_core.runnables import RunnableConfig
# from trustcall import create_extractor
# from langgraph.store.base import BaseStore
# from langchain_core.messages import merge_message_runs
# import uuid
# from datetime import datetime
# from typing import Literal

# # Import local modules
# import configuration
# from models import model, MODEL_SYSTEM_MESSAGE, TRUSTCALL_INSTRUCTION, profile_extractor
# from profile import ToDo, Profile, UpdateMemory
# from spy import Spy, extract_tool_info

# # Import search functionality
# try:
#     from search_integration import perform_search, format_search_results, should_search
#     SEARCH_AVAILABLE = True
# except ImportError:
#     SEARCH_AVAILABLE = False


# def task_manager(state: MessagesState, config: RunnableConfig):
#     """Main task manager function that processes user messages and decides on actions."""
    
#     # Get store from config - check both places it might be
#     store = config.get("store") or config.get("configurable", {}).get("store")
#     if not store:
#         # Print the structure of config for debugging
#         print(f"Config keys: {list(config.keys())}")
#         print(f"Configurable keys: {list(config.get('configurable', {}).keys())}")
#         raise ValueError("Store not found in config")
    
#     # Get user ID from config
#     configurable = configuration.Configuration.from_runnable_config(config)
#     user_id = configurable.user_id
#     todo_category = configurable.todo_category
#     task_manager_role = configurable.task_manager_role

#     # Check for task list query
#     last_message = state['messages'][-1]
#     last_message_content = last_message.content if hasattr(last_message, 'content') else ""
    
#     # Handle search requests
#     if SEARCH_AVAILABLE and isinstance(last_message_content, str) and is_search_request(last_message_content):
#         # Get user profile for context
#         namespace = ('profile', todo_category, user_id)
#         profiles = store.search(namespace)
#         user_profile = profiles[0].value if profiles else None
        
#         # Get tasks for context
#         namespace = ('todo', todo_category, user_id)
#         tasks = store.search(namespace)
#         todo_list = '\n'.join(f'• {task.value.get("task", "Unknown task")}' for task in tasks)
        
#         # Handle "yes" responses to search offers
#         previous_msg = state['messages'][-2].content if len(state['messages']) > 1 and hasattr(state['messages'][-2], 'content') else ""
#         if last_message_content.lower() in ["yes", "sure", "okay", "please do", "go ahead"] and "would you like me to" in previous_msg.lower() and "search" in previous_msg.lower():
#             # Extract search query from previous message
#             import re
#             match = re.search(r"look up (.*?)( for you)?(\?|\.)", previous_msg.lower())
#             search_query = match.group(1) if match else previous_msg
#         else:
#             search_query = last_message_content
        
#         # Perform search
#         try:
#             search_results = perform_search(search_query)
#             if "error" not in search_results:
#                 search_response = format_search_results(model, search_query, search_results, todo_list)
#                 return {'messages': [AIMessage(content=search_response)]}
#         except Exception as e:
#             print(f"Search error: {str(e)}")
#             # Fall back to regular processing if search fails
    
#     # Check for task removal
#     if isinstance(last_message_content, str) and is_task_removal_message(last_message_content):
#         # Handle task removal directly here instead of in a separate node
#         removal_response = handle_task_removal(last_message_content, store, configurable, model)
#         if removal_response:
#             return {'messages': [AIMessage(content=removal_response)]}
    
#     # Check for task list query
#     elif isinstance(last_message_content, str) and any(query in last_message_content.lower() for query in ["my tasks", "my to do", "todo list", "what are my", "what do i have", "show me my"]):
#         namespace = ('todo', todo_category, user_id)
#         tasks = store.search(namespace)
        
#         if tasks:
#             task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')} (Est: {task.value.get('time_to_complete', '?')} mins)" 
#                                  for task in tasks])
#             response = AIMessage(content=f"Here's your current task list:\n\n{task_list}")
#             return {'messages': [response]}
#         else:
#             response = AIMessage(content="You don't have any tasks yet. Would you like to add some?")
#             return {'messages': [response]}

#     # Process standard task addition
#     elif isinstance(last_message_content, str) and looks_like_task(last_message_content) and not is_search_request(last_message_content):
#         namespace = ('todo', todo_category, user_id)
        
#         # Create a new simple task
#         todo_data = {
#             "task": last_message_content,
#             "time_to_complete": 60,  # Default time
#             "solutions": ["Complete this task"],
#             "status": "not started"
#         }
        
#         # Add to store
#         task_id = str(uuid.uuid4())
#         store.put(namespace, task_id, todo_data)
        
#         # Return confirmation
#         return {'messages': [AIMessage(content=f"I've added the task \"{last_message_content}\" to your list.")]}

#     # Get profile from memory state
#     namespace = ('profile', todo_category, user_id)
#     memories = store.search(namespace)
#     if memories:
#         user_profile = memories[0].value
#     else:
#         user_profile = None

#     # Get tasks from memory state
#     namespace = ('todo', todo_category, user_id)
#     memories = store.search(namespace)
#     todo = '\n'.join(f'{mem.value}' for mem in memories)

#     # Get instructions from custom setting
#     namespace = ('instructions', todo_category, user_id)
#     memories = store.search(namespace)
#     if memories:
#         instructions = memories[0].value
#     else:
#         instructions = ''

#     # Format system message with user profile, todo list, and instructions
#     system_msg = MODEL_SYSTEM_MESSAGE.format(
#         task_manager_role=task_manager_role, 
#         user_profile=user_profile, 
#         todo=todo, 
#         instructions=instructions
#     )

#     # Generate response with model
#     response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
#         [SystemMessage(content=system_msg)] + state['messages']
#     )

#     return {'messages': [response]}


# def is_search_request(text):
#     """Determine if text is a search request."""
#     if not isinstance(text, str):
#         return False
        
#     lower_text = text.lower()
    
#     # Common location/search phrases
#     search_patterns = [
#         "where is", "find", "search for", "look up", "where can i",
#         "nearest", "closest", "directions to", "how far is", "location of",
#         "address for", "address of", "google", "search", "map"
#     ]
    
#     # Check for question about locations or businesses
#     if any(pattern in lower_text for pattern in search_patterns):
#         # Verify it's not a task by checking for task indicators
#         task_indicators = ["i need to", "add", "remind me to", "put on my list"]
#         if not any(indicator in lower_text for indicator in task_indicators):
#             return True
            
#     # Also handle yes/no responses to search offers
#     if lower_text in ["yes", "sure", "okay", "please do", "go ahead"]:
#         return True
            
#     return False


# def looks_like_task(text):
#     """Check if a message looks like a task to be added."""
#     if not isinstance(text, str):
#         return False
        
#     lower_text = text.lower()
#     task_indicators = ["i need", "i want", "i should", "i must", "need to", 
#                       "going to", "have to", "should", "must", "will"]
    
#     # Check if it has task indicators and doesn't look like a question or search
#     if any(indicator in lower_text for indicator in task_indicators) and not any(
#         keyword in lower_text for keyword in ["where", "how", "what is", "who is", 
#                                           "when", "why", "find", "search", "look up"]) and "?" not in lower_text:
#         return True
    
#     return False


# def is_task_removal_message(text):
#     """Check if a message indicates task completion or removal."""
#     if not isinstance(text, str):
#         return False
        
#     lower_text = text.lower()
#     removal_indicators = [
#         "remove", "delete", "complete", "mark as done", "done", "finished",
#         "completed", "remove that task", "take off", "cross off", "clear",
#         "i bought", "i purchased", "i got", "i did", "i finished"
#     ]
    
#     return any(indicator in lower_text for indicator in removal_indicators)


# def handle_task_removal(text, store, config, model_instance):
#     """Remove tasks based on user message."""
    
#     # Get namespace for todo items
#     namespace = ('todo', config.todo_category, config.user_id)
#     tasks = store.search(namespace)
    
#     if not tasks:
#         return "You don't have any tasks to remove."
    
#     # Create system message to analyze which tasks to remove
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
#             task_text = task.value.get('task', 'Unknown task')
#             store.remove(namespace, task.key)
#             removed_tasks.append(task_text)
#     elif result != "NONE":
#         try:
#             indices = [int(idx.strip()) - 1 for idx in result.split(',')]
#             # Sort indices in descending order to avoid index shifting when removing items
#             for idx in sorted(indices, reverse=True):
#                 if 0 <= idx < len(tasks):
#                     task = tasks[idx]
#                     task_text = task.value.get('task', 'Unknown task')
#                     store.remove(namespace, task.key)
#                     removed_tasks.append(task_text)
#         except ValueError:
#             pass
    
#     # Verify removal was successful
#     updated_tasks = store.search(namespace)
#     remaining_task_texts = [task.value.get('task', '') for task in updated_tasks]
    
#     # Generate response
#     if removed_tasks:
#         if len(removed_tasks) == 1:
#             return f"I've removed the task: \"{removed_tasks[0]}\" from your list."
#         else:
#             task_list = "\n".join([f"• {task}" for task in removed_tasks])
#             return f"I've removed these tasks from your list:\n{task_list}"
    
#     # If no tasks were identified for removal
#     return "I couldn't identify which task you wanted to remove. Please specify which task you've completed."


# def update_profile(state: MessagesState, config: RunnableConfig):
#     """Update profile information based on conversation history."""
    
#     # Get store from config - check both places it might be
#     store = config.get("store") or config.get("configurable", {}).get("store")
#     if not store:
#         raise ValueError("Store not found in config")
    
#     # Get user ID from the config
#     configurable = configuration.Configuration.from_runnable_config(config)
#     user_id = configurable.user_id
#     todo_category = configurable.todo_category

#     # Define the namespace for the memories
#     namespace = ('profile', todo_category, user_id)

#     # Retrieve the most recent memories for context
#     existing_items = store.search(namespace)

#     # Format the existing memories for the Trustcall extractor
#     tool_name = 'Profile'
#     existing_memories = (
#         [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
#         if existing_items
#         else None
#     )

#     # Merge chat history and instructions
#     TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
#     updated_messages = list(merge_message_runs(
#         messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]
#     ))

#     # Invoke profile extractor
#     result = profile_extractor.invoke({
#         'messages': updated_messages, 
#         'existing': existing_memories
#     })

#     # Save the memories from Trustcall to the store
#     for r, rmeta in zip(result['responses'], result['response_metadata']):
#         store.put(
#             namespace, 
#             rmeta.get('json_doc_id', str(uuid.uuid4())),
#             r.model_dump(mode='json') if hasattr(r, 'model_dump') else r,
#         )

#     # Get tool call ID for response
#     tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
#     tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

#     # Return tool response
#     return {'messages': [{'role': 'tool', 'content': 'updated profile', 'tool_call_id': tool_call_id}]}


# def update_todos(state: MessagesState, config: RunnableConfig):
#     """Update task list based on conversation history."""
    
#     # Get store from config - check both places it might be
#     store = config.get("store") or config.get("configurable", {}).get("store")
#     if not store:
#         raise ValueError("Store not found in config")
    
#     # Get user ID
#     configurable = configuration.Configuration.from_runnable_config(config)
#     user_id = configurable.user_id
#     todo_category = configurable.todo_category

#     # Define the namespace for the memories
#     namespace = ('todo', todo_category, user_id)    

#     # Retrieve most recent memories
#     existing_items = store.search(namespace)

#     # Get the last message
#     last_message = state['messages'][-2] if len(state['messages']) > 1 else state['messages'][-1]
#     content = last_message.content if hasattr(last_message, 'content') else ""
    
#     # Skip task creation if it looks like a search query
#     if is_search_request(content):
#         # Just return without adding a task
#         tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
#         tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"
#         return {'messages': [{'role': 'tool', 'content': 'No task added - this appears to be a search query', 'tool_call_id': tool_call_id}]}
    
#     # If it looks like a task (not a question or simple reply)
#     if isinstance(content, str) and not content.lower() in ["yes", "no", "ok", "sure", "thanks", "thank you"] and not content.lower().startswith(("what", "where", "when", "how", "why", "can you", "could you")) and "?" not in content:
#         # Create a new simple task
#         todo_data = {
#             "task": content,
#             "time_to_complete": 60,  # Default time
#             "solutions": ["Complete this task"],
#             "status": "not started"
#         }
        
#         # Add to store
#         task_id = str(uuid.uuid4())
#         store.put(namespace, task_id, todo_data)
    
#     # Format the existing memories for the Trustcall
#     tool_name = 'ToDo'
#     existing_memories = (
#         [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
#         if existing_items
#         else None
#     )

#     # Merge chat
#     TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
#     updated_messages = list(merge_message_runs(
#         messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]
#     ))

#     # Initialize spy
#     spy = Spy()

#     # Create todo extractor
#     todo_extractor = create_extractor(
#         model,
#         tools=[ToDo],
#         tool_choice=tool_name,
#         enable_inserts=True
#     ).with_listeners(on_end=spy)

#     # Invoke extractor
#     result = todo_extractor.invoke({
#         'messages': updated_messages, 
#         'existing': existing_memories
#     })

#     # Save memories from trustcall to store
#     for r, rmeta in zip(result['responses'], result['response_metadata']):
#         store.put(
#             namespace, 
#             rmeta.get('json_doc_id', str(uuid.uuid4())),
#             r.model_dump(mode='json') if hasattr(r, 'model_dump') else r,
#         )

#     # Get tool call ID for response
#     tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
#     tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

#     # Return tool response
#     return {'messages': [{'role': 'tool', 'content': f"I've added this task to your list:\n• {content}", 'tool_call_id': tool_call_id}]}


# def update_instructions(state: MessagesState, config: RunnableConfig):
#     """Update instructions based on user preferences."""
    
#     # Get store from config - check both places it might be
#     store = config.get("store") or config.get("configurable", {}).get("store")
#     if not store:
#         raise ValueError("Store not found in config")
    
#     # Get user id
#     configurable = configuration.Configuration.from_runnable_config(config)
#     user_id = configurable.user_id 
#     todo_category = configurable.todo_category

#     # Get existing instructions
#     namespace = ('instructions', todo_category, user_id)
#     existing_memory = store.search(namespace, 'user_instructions')

#     # Create system message with current instructions
#     from models import CREATE_INSTRUCTIONS
#     system_msg = CREATE_INSTRUCTIONS.format(
#         current_instructions=existing_memory[0].value if existing_memory else None
#     )
    
#     # Generate new instructions
#     new_memory = model.invoke(
#         [SystemMessage(content=system_msg)] + 
#         [state['messages'][-1]] + 
#         [HumanMessage(content='Please update the instructions based on the conversation')]
#     )

#     # Save new instructions
#     key = 'user_instructions'
#     store.put(namespace, key, new_memory.content)
    
#     # Get tool call ID for response
#     tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
#     tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

#     # Return tool response
#     return {'messages': [{'role': 'tool', 'content': 'updated instructions', 'tool_call_id': tool_call_id}]}


# def route_message(state: MessagesState, config: RunnableConfig) -> Literal[END, 'update_todos', 'update_instructions', 'update_profile']:
#     """Route messages to appropriate handlers based on tool calls."""
    
#     message = state['messages'][-1]
    
#     # If no tool calls, end the flow
#     if not hasattr(message, 'tool_calls') or not message.tool_calls:
#         return END
    
#     # Otherwise route based on tool call
#     tool_call = message.tool_calls[0]
#     if 'args' in tool_call and 'update_type' in tool_call['args']:
#         update_type = tool_call['args']['update_type']
        
#         if update_type == "user":
#             return 'update_profile'
#         elif update_type == "todo":
#             return 'update_todos'
#         elif update_type == "instructions":
#             return 'update_instructions'
    
#     # Default to END if no match
#     return END

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables import RunnableConfig
from trustcall import create_extractor
from langgraph.store.base import BaseStore
from langchain_core.messages import merge_message_runs
import uuid
from datetime import datetime
from typing import Literal

# Import local modules
import configuration
from models import model, MODEL_SYSTEM_MESSAGE, TRUSTCALL_INSTRUCTION, profile_extractor
from profile import ToDo, Profile, UpdateMemory
from spy import Spy, extract_tool_info

# Import search functionality (if available)
try:
    from search_integration import perform_search, format_search_results
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


def determine_message_intent(text, model_instance):
    """Use the LLM to determine the intent of the message."""
    
    system_message = """
    Analyze the user's message and determine its intent.
    Respond with ONLY ONE of the following categories:
    - SEARCH: User is asking for information that requires searching or looking up information
    - TASK_ADD: User wants to add a task or is stating something they need to do
    - TASK_REMOVE: User wants to remove, complete, or mark a task as done
    - TASK_LIST: User wants to see their task list
    - CONVERSATION: General conversation that doesn't fit the above categories
    
    Return ONLY the category label with no additional text.
    """
    
    response = model_instance.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"User message: {text}")
    ])
    
    result = response.content.strip() if hasattr(response, 'content') else str(response)
    return result


def task_manager(state: MessagesState, config: RunnableConfig):
    """Main task manager function that processes user messages and determines intent."""
    
    # Get store from config
    store = config.get("store") or config.get("configurable", {}).get("store")
    if not store:
        raise ValueError("Store not found in config")
    
    # Get user ID from config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_manager_role = configurable.task_manager_role

    # Get the last message
    last_message = state['messages'][-1]
    last_message_content = last_message.content if hasattr(last_message, 'content') else ""
    
    # Skip processing if empty message
    if not isinstance(last_message_content, str) or not last_message_content.strip():
        return {'messages': [AIMessage(content="I didn't receive any message. How can I help you?")]}
    
    # Determine intent using LLM
    intent = determine_message_intent(last_message_content, model)
    
    # Handle search intent
    if intent == "SEARCH" and SEARCH_AVAILABLE:
        # Handle "yes" to search with context from previous message
        previous_msg = state['messages'][-2].content if len(state['messages']) > 1 and hasattr(state['messages'][-2], 'content') else ""
        search_query = last_message_content
        
        # For affirmative responses to search offers, extract the search query
        if search_query.lower() in ["yes", "sure", "okay", "please do", "go ahead"] and "would you like me to" in previous_msg.lower():
            import re
            match = re.search(r"look up (.*?)( for you)?(\?|\.)", previous_msg.lower())
            if match:
                search_query = match.group(1)
            else:
                # Try to extract from between quotes
                match = re.search(r'"([^"]*)"', previous_msg)
                search_query = match.group(1) if match else previous_msg
        
        # Add context for search
        namespace = ('profile', todo_category, user_id)
        profiles = store.search(namespace)
        user_profile = profiles[0].value if profiles else None
        
        namespace = ('todo', todo_category, user_id)
        tasks = store.search(namespace)
        todo_list = '\n'.join(f'• {task.value.get("task", "Unknown task")}' for task in tasks)
        
        # Perform search
        try:
            search_results = perform_search(search_query)
            if "error" not in search_results:
                search_response = format_search_results(model, search_query, search_results, todo_list)
                return {'messages': [AIMessage(content=search_response)]}
            else:
                return {'messages': [AIMessage(content=f"I tried to search for '{search_query}' but encountered an error: {search_results['error']}")]}
        except Exception as e:
            return {'messages': [AIMessage(content=f"I wasn't able to perform the search. {str(e)}")]}
    
    # Handle task list intent
    elif intent == "TASK_LIST":
        namespace = ('todo', todo_category, user_id)
        tasks = store.search(namespace)
        
        if tasks:
            task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')} (Est: {task.value.get('time_to_complete', '?')} mins)" 
                                 for task in tasks])
            response = AIMessage(content=f"Here is your current to-do list:\n{task_list}")
            return {'messages': [response]}
        else:
            response = AIMessage(content="You don't have any tasks yet. Would you like to add some?")
            return {'messages': [response]}
    
    # Handle task removal intent
    elif intent == "TASK_REMOVE":
        # Get tasks
        namespace = ('todo', todo_category, user_id)
        tasks = store.search(namespace)
        
        if not tasks:
            return {'messages': [AIMessage(content="You don't have any tasks to remove.")]}
        
        # Ask LLM which tasks to remove
        system_message = """
        You are analyzing a user request to remove or complete tasks from a todo list.
        Given the user's message and their current task list, identify which task(s) should be removed.
        
        Return ONLY the index numbers (starting from 1) of tasks to remove, separated by commas.
        If all tasks should be removed, return "ALL".
        If no specific task can be identified, return "NONE".
        """
        
        task_list = "\n".join([f"{i+1}. {task.value.get('task', 'Unknown task')}" 
                             for i, task in enumerate(tasks)])
        
        response = model.invoke([
            SystemMessage(content=system_message),
            HumanMessage(content=f"User message: {last_message_content}\n\nCurrent tasks:\n{task_list}")
        ])
        
        result = response.content.strip() if hasattr(response, 'content') else str(response)
        
        removed_tasks = []
        if result == "ALL":
            for task in list(tasks):
                task_text = task.value.get('task', 'Unknown task')
                store.remove(namespace, task.key)
                removed_tasks.append(task_text)
        elif result != "NONE":
            try:
                indices = [int(idx.strip()) - 1 for idx in result.split(',')]
                for idx in sorted(indices, reverse=True):
                    if 0 <= idx < len(tasks):
                        task = tasks[idx]
                        task_text = task.value.get('task', 'Unknown task')
                        store.remove(namespace, task.key)
                        removed_tasks.append(task_text)
            except ValueError:
                pass
        
        if removed_tasks:
            if len(removed_tasks) == 1:
                return {'messages': [AIMessage(content=f"I've removed the task: \"{removed_tasks[0]}\" from your list.")]}
            else:
                task_list = "\n".join([f"• {task}" for task in removed_tasks])
                return {'messages': [AIMessage(content=f"I've removed these tasks from your list:\n{task_list}")]}
        else:
            return {'messages': [AIMessage(content="I couldn't identify which task you wanted to remove. Please specify which task you've completed.")]}
    
    # Handle task addition intent
    elif intent == "TASK_ADD":
        namespace = ('todo', todo_category, user_id)
        
        # Create a new task
        todo_data = {
            "task": last_message_content,
            "time_to_complete": 60,
            "solutions": ["Complete this task"],
            "status": "not started"
        }
        
        # Add to store
        task_id = str(uuid.uuid4())
        store.put(namespace, task_id, todo_data)
        
        # Return confirmation
        return {'messages': [AIMessage(content=f"I've added the task \"{last_message_content}\" to your list.")]}
    
    # For general conversation or if intent classification failed
    # Get profile from memory state
    namespace = ('profile', todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Get tasks from memory state
    namespace = ('todo', todo_category, user_id)
    memories = store.search(namespace)
    todo = '\n'.join(f'{mem.value}' for mem in memories)

    # Get instructions from custom setting
    namespace = ('instructions', todo_category, user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ''

    # Format system message with user profile, todo list, and instructions
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        task_manager_role=task_manager_role, 
        user_profile=user_profile, 
        todo=todo, 
        instructions=instructions
    )

    # Generate response with model
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state['messages']
    )

    return {'messages': [response]}


def update_profile(state: MessagesState, config: RunnableConfig):
    """Update profile information based on conversation history."""
    
    # Get store from config
    store = config.get("store") or config.get("configurable", {}).get("store")
    if not store:
        raise ValueError("Store not found in config")
    
    # Get user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Define the namespace for the memories
    namespace = ('profile', todo_category, user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = 'Profile'
    existing_memories = (
        [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge chat history and instructions
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]
    ))

    # Invoke profile extractor
    result = profile_extractor.invoke({
        'messages': updated_messages, 
        'existing': existing_memories
    })

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result['responses'], result['response_metadata']):
        store.put(
            namespace, 
            rmeta.get('json_doc_id', str(uuid.uuid4())),
            r.model_dump(mode='json') if hasattr(r, 'model_dump') else r,
        )

    # Get tool call ID for response
    tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
    tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

    # Return tool response
    return {'messages': [{'role': 'tool', 'content': 'updated profile', 'tool_call_id': tool_call_id}]}


def update_todos(state: MessagesState, config: RunnableConfig):
    """Update task list based on conversation history."""
    
    # Get store from config
    store = config.get("store") or config.get("configurable", {}).get("store")
    if not store:
        raise ValueError("Store not found in config")
    
    # Get user ID
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category

    # Define the namespace for the memories
    namespace = ('todo', todo_category, user_id)    

    # Retrieve most recent memories
    existing_items = store.search(namespace)

    # Get the last message
    last_message = state['messages'][-2] if len(state['messages']) > 1 else state['messages'][-1]
    content = last_message.content if hasattr(last_message, 'content') else ""
    
    # Check intent before adding task
    intent = determine_message_intent(content, model)
    if intent != "TASK_ADD":
        # If not a task, don't add it
        tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
        tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"
        return {'messages': [{'role': 'tool', 'content': 'No task added - not a task request', 'tool_call_id': tool_call_id}]}
    
    # Create a new task
    todo_data = {
        "task": content,
        "time_to_complete": 60,  # Default time
        "solutions": ["Complete this task"],
        "status": "not started"
    }
    
    # Add to store
    task_id = str(uuid.uuid4())
    store.put(namespace, task_id, todo_data)
    
    # Format the existing memories for the Trustcall
    tool_name = 'ToDo'
    existing_memories = (
        [(existing_item.key, tool_name, existing_item.value) for existing_item in existing_items]
        if existing_items
        else None
    )

    # Merge chat
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(
        messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state['messages'][:-1]
    ))

    # Initialize spy
    spy = Spy()

    # Create todo extractor
    todo_extractor = create_extractor(
        model,
        tools=[ToDo],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke extractor
    result = todo_extractor.invoke({
        'messages': updated_messages, 
        'existing': existing_memories
    })

    # Save memories from trustcall to store
    for r, rmeta in zip(result['responses'], result['response_metadata']):
        store.put(
            namespace, 
            rmeta.get('json_doc_id', str(uuid.uuid4())),
            r.model_dump(mode='json') if hasattr(r, 'model_dump') else r,
        )

    # Get tool call ID for response
    tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
    tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

    # Return tool response
    return {'messages': [{'role': 'tool', 'content': f"I've added this task to your list:\n• {content}", 'tool_call_id': tool_call_id}]}


def update_instructions(state: MessagesState, config: RunnableConfig):
    """Update instructions based on user preferences."""
    
    # Get store from config
    store = config.get("store") or config.get("configurable", {}).get("store")
    if not store:
        raise ValueError("Store not found in config")
    
    # Get user id
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id 
    todo_category = configurable.todo_category

    # Get existing instructions
    namespace = ('instructions', todo_category, user_id)
    existing_memory = store.search(namespace, 'user_instructions')

    # Create system message with current instructions
    from models import CREATE_INSTRUCTIONS
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory[0].value if existing_memory else None
    )
    
    # Generate new instructions
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)] + 
        [state['messages'][-1]] + 
        [HumanMessage(content='Please update the instructions based on the conversation')]
    )

    # Save new instructions
    key = 'user_instructions'
    store.put(namespace, key, new_memory.content)
    
    # Get tool call ID for response
    tool_calls = state['messages'][-1].tool_calls if hasattr(state['messages'][-1], 'tool_calls') else []
    tool_call_id = tool_calls[0]['id'] if tool_calls else "tool_call_1"

    # Return tool response
    return {'messages': [{'role': 'tool', 'content': 'updated instructions', 'tool_call_id': tool_call_id}]}


def route_message(state: MessagesState, config: RunnableConfig) -> Literal[END, 'update_todos', 'update_instructions', 'update_profile']:
    """Route messages to appropriate handlers based on tool calls."""
    
    message = state['messages'][-1]
    
    # If no tool calls, end the flow
    if not hasattr(message, 'tool_calls') or not message.tool_calls:
        return END
    
    # Otherwise route based on tool call
    tool_call = message.tool_calls[0]
    if 'args' in tool_call and 'update_type' in tool_call['args']:
        update_type = tool_call['args']['update_type']
        
        if update_type == "user":
            return 'update_profile'
        elif update_type == "todo":
            return 'update_todos'
        elif update_type == "instructions":
            return 'update_instructions'
    
    # Default to END if no match
    return END
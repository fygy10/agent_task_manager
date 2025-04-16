from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
import uuid


from set_env import setup_environment
from models import MODEL_SYSTEM_MESSAGE, model
from configuration import Configuration
from profile import UpdateMemory, ToDo, Profile
from trustcall import create_extractor
from search_integration import handle_search_request


#in-memory storage
#pass in tuple(data type, category, user ID) - ex. (ToDO, work, ID 1)
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
        print(f"Stored data in {namespace_str}/{key}")

        # Add this method to your SimpleStore class in main.py

    def remove(self, namespace, key):
        """
        Remove an item from the store.
        
        Args:
            namespace (tuple): The namespace tuple
            key (str): The key to remove
            
        Returns:
            bool: True if item was removed, False otherwise
        """
        namespace_str = "_".join(str(part) for part in namespace)
        if namespace_str not in self.data:
            return False
        
        if key not in self.data[namespace_str]:
            return False
        
        del self.data[namespace_str][key]
        print(f"Removed data from {namespace_str}/{key}")
        return True


def determine_input_type(model_instance, text):
    """
    Use the LLM to determine if the input is a question, task, or information sharing.
    This replaces hardcoded pattern matching with LLM intelligence.
    """
    system_message = """
    Analyze the user's input and classify it as one of the following types:
    1. "question" - The user is asking for information or recommendations
    2. "task" - The user is describing a task they want to complete or add to their list
    3. "conversation" - The user is making a simple conversational response
    
    Reply ONLY with one of these three words based on your classification.
    """
    
    response = model_instance.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=text)
    ])
    
    content = response.content.lower().strip() if hasattr(response, 'content') else ""
    
    if "question" in content:
        return "question"
    elif "task" in content:
        return "task"
    else:
        return "conversation"


def process_user_input(model_instance, store, user_input, config):
    """Process user input using the LLM-based extractor."""
    # Create extractors for both Profile and ToDo with the fixed create_extractor function
    profile_extractor = create_extractor(
        model=model_instance, 
        tools=[Profile], 
        tool_choice="Profile"
    )
    
    todo_extractor = create_extractor(
        model=model_instance,
        tools=[ToDo],
        tool_choice="ToDo"
    )
    
    # Determine input type using the LLM
    input_type = determine_input_type(model_instance, user_input)
    
    # Create a message object from user input
    messages = [HumanMessage(content=user_input)]
    
    # Get namespaces
    namespace_profile = ('profile', config.todo_category, config.user_id)
    namespace_todo = ('todo', config.todo_category, config.user_id)
    
    # Always extract profile information regardless of input type
    # This allows the system to update the user profile based on any conversation
    existing_profiles = store.search(namespace_profile)
    existing_profile_data = None
    if existing_profiles:
        existing_profile_data = [(item.key, "Profile", item.value) for item in existing_profiles]
    
    # Extract profile information
    profile_result = profile_extractor.invoke({
        'messages': messages,
        'existing': existing_profile_data
    })
    
    # Save any extracted profile information
    for r, rmeta in zip(profile_result.get('responses', []), profile_result.get('response_metadata', [])):
        if hasattr(r, 'model_dump'):
            store.put(namespace_profile, rmeta.get('json_doc_id', str(uuid.uuid4())), r.model_dump(mode='json'))
        else:
            store.put(namespace_profile, rmeta.get('json_doc_id', str(uuid.uuid4())), r)
    
    # If this is a task, process it with the todo extractor
    if input_type == "task":
        # Get existing todos
        existing_todos = store.search(namespace_todo)
        existing_todo_data = None
        if existing_todos:
            existing_todo_data = [(item.key, "ToDo", item.value) for item in existing_todos]
        
        # Extract todo information
        todo_result = todo_extractor.invoke({
            'messages': messages,
            'existing': existing_todo_data
        })
        
        # Process and save extracted todos
        tasks_added = []
        for r, rmeta in zip(todo_result.get('responses', []), todo_result.get('response_metadata', [])):
            if hasattr(r, 'model_dump'):
                task_data = r.model_dump(mode='json')
            else:
                task_data = r
            
            # Add to store
            task_id = rmeta.get('json_doc_id', str(uuid.uuid4()))
            store.put(namespace_todo, task_id, task_data)
            tasks_added.append(task_data.get('task', 'Unknown task'))
        
        # If we extracted tasks, report them
        if tasks_added:
            task_list = "\n".join([f"• {task}" for task in tasks_added])
            return {
                "type": "task",
                "tasks_added": tasks_added,
                "response": f"I've added these tasks to your list:\n{task_list}"
            }
        
        # If no tasks were extracted but input was classified as a task,
        # create a simple task from the user input
        task_title = user_input
        if len(task_title) > 100:
            task_title = task_title[:97] + "..."
        
        task_data = {
            "task": task_title,
            "time_to_complete": 60,
            "solutions": ["Complete this task"],
            "status": "not started"
        }
        
        # Add to store
        task_id = str(uuid.uuid4())
        store.put(namespace_todo, task_id, task_data)
        
        return {
            "type": "task",
            "tasks_added": [task_title],
            "response": f"I've added the task \"{task_title}\" to your list."
        }
    
    # For questions or conversation, get the profile and tasks for context
    # but let the LLM generate a response
    return {
        "type": input_type,
        "response": None  # Will be handled by LLM response
    }


def get_llm_response(model_instance, messages, store, config):
    """Generate a response using the LLM with context from the store."""
    # Get profile from memory
    namespace = ('profile', config.todo_category, config.user_id)
    profile_memories = store.search(namespace)
    user_profile = profile_memories[0].value if profile_memories else None

    # Get tasks from memory
    namespace = ('todo', config.todo_category, config.user_id)
    task_memories = store.search(namespace)
    todo_list = '\n'.join(f'• {mem.value.get("task", "Unknown task")}' for mem in task_memories)

    # Get instructions from custom setting
    namespace = ('instructions', config.todo_category, config.user_id)
    instruction_memories = store.search(namespace)
    instructions = instruction_memories[0].value if instruction_memories else ''

    # Format system message with context
    system_msg = MODEL_SYSTEM_MESSAGE.format(
        task_manager_role=config.task_manager_role, 
        user_profile=user_profile, 
        todo=todo_list, 
        instructions=instructions
    )

    # Generate response
    response = model_instance.invoke(
        [SystemMessage(content=system_msg)] + messages
    )

    return response.content if hasattr(response, 'content') else str(response)

def handle_task_removal(text, store, config):
    """
    Handle requests to remove tasks from the list.
    
    Args:
        text (str): User input text
        store: The store instance
        config: The configuration object
        
    Returns:
        str: Response message if tasks were removed, None otherwise
    """
    # Check if this is a task removal request
    lower_text = text.lower()
    removal_indicators = [
        "remove", "delete", "complete", "mark as done", "done", "finished",
        "completed", "remove that task", "take off", "cross off", "clear"
    ]
    
    is_removal_request = any(indicator in lower_text for indicator in removal_indicators)
    
    if not is_removal_request:
        return None
    
    # Get current tasks
    namespace = ('todo', config.todo_category, config.user_id)
    tasks = store.search(namespace)
    
    if not tasks:
        return "You don't have any tasks to remove."
    
    # Use an LLM to identify which task(s) should be removed
    system_message = """
    You are analyzing a user request to remove or complete tasks from a todo list.
    Given the user's message and their current task list, identify which task(s) should be removed.
    
    Return ONLY the index numbers (starting from 1) of tasks to remove, separated by commas.
    If all tasks should be removed, return "ALL".
    If no specific task can be identified, return "NONE".
    """
    
    task_list = "\n".join([f"{i+1}. {task.value.get('task', 'Unknown task')}" 
                         for i, task in enumerate(tasks)])
    
    from langchain_core.messages import SystemMessage, HumanMessage
    from models import model
    
    response = model.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"User message: {text}\n\nCurrent tasks:\n{task_list}")
    ])
    
    result = response.content.strip() if hasattr(response, 'content') else str(response)
    
    # Process the result
    if result == "NONE":
        return None
    
    removed_tasks = []
    if result == "ALL":
        # Remove all tasks
        for task in tasks:
            store.remove(namespace, task.key)
            removed_tasks.append(task.value.get('task', 'Unknown task'))
    else:
        try:
            # Parse indices
            indices = [int(idx.strip()) - 1 for idx in result.split(',')]
            for idx in indices:
                if 0 <= idx < len(tasks):
                    task = tasks[idx]
                    store.remove(namespace, task.key)
                    removed_tasks.append(task.value.get('task', 'Unknown task'))
        except ValueError:
            return None
    
    if removed_tasks:
        if len(removed_tasks) == 1:
            return f"I've removed the task: \"{removed_tasks[0]}\" from your list."
        else:
            task_list = "\n".join([f"• {task}" for task in removed_tasks])
            return f"I've removed these tasks from your list:\n{task_list}"
    
    return None

def main():
    """Run a task manager application interactively with LLM-based extraction."""
    # Initialize the LLM
    model_instance = setup_environment()
    
    # Create simple store
    store = SimpleStore()
    
    # Create configuration
    config = Configuration(
        user_id="user_123",
        todo_category="personal",
        task_manager_role="You are an AI assistant that helps users organize their tasks and manage their to-do lists. You are friendly, helpful, and efficient."
    )
    
    print("\n💼 Task Manager Assistant 💼")
    print("─────────────────────────────")
    print("Hi! I'm your task management assistant. How can I help you organize your tasks today?")
    print("─────────────────────────────")
    
    # Message history for context
    messages = []
    
    # Interactive loop
    while True:
        # Get user input
        user_input = input("\n> ")
        
        # Exit if requested
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye! Have a productive day!")
            break
        
        # Skip empty inputs
        if not user_input.strip():
            continue
        
        # Create user message
        message = HumanMessage(content=user_input)
        messages.append(message)
        
        #Get user profile information for search context
        namespace_profile = ('profile', config.todo_category, config.user_id)
        profile_memories = store.search(namespace_profile)
        search_context = {}

        for mem in profile_memories:
            if isinstance(mem.value, dict):
                if "location" in mem.value and mem.value["location"]:
                    search_context["location"] = mem.value["location"]

        # Get task list for context
        namespace = ('todo', config.todo_category, config.user_id)
        task_memories = store.search(namespace)
        todo_list = '\n'.join(f'• {mem.value.get("task", "Unknown task")}' for mem in task_memories)

        # Check if this should be a search request
        search_response = handle_search_request(model_instance, user_input, search_context, todo_list)

        if search_response:
            print(f"\n💼 {search_response}")
            # Add AI response to message history
            messages.append(AIMessage(content=search_response))
            continue
        
        # Handle task list query directly for efficiency
        if any(query in user_input.lower() for query in ["my tasks", "my to do", "todo list", "what are my", "what do i have", "show me my"]):
            namespace = ('todo', config.todo_category, config.user_id)
            tasks = store.search(namespace)
            
            if tasks:
                task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')}" 
                                     for task in tasks])
                print(f"\n💼 Here's your current task list:\n\n{task_list}")
            else:
                print("\n💼 You don't have any tasks yet. Would you like to add some?")
            
            continue


        # Handle task removal requests
        removal_response = handle_task_removal(user_input, store, config)
        if removal_response:
            print(f"\n💼 {removal_response}")
            # Add the response to message history for context
            messages.append(AIMessage(content=removal_response))
            continue

        # Get task list for context
        namespace = ('todo', config.todo_category, config.user_id)
        task_memories = store.search(namespace)
        todo_list = '\n'.join(f'• {mem.value.get("task", "Unknown task")}' for mem in task_memories)

        # Check if this should be a search request
        search_response = handle_search_request(model_instance, user_input, search_context, todo_list)
        
        # Process user input with the LLM-based extractor
        result = process_user_input(model_instance, store, user_input, config)
        
        # If the extractor generated a response, display it
        if result and result.get("response"):
            print(f"\n💼 {result['response']}")
            # Add the response to message history for context
            messages.append(AIMessage(content=result['response']))
        else:
            # Otherwise, generate a response using the LLM
            llm_response = get_llm_response(model_instance, messages, store, config)
            print(f"\n💼 {llm_response}")
            # Add the response to message history for context
            messages.append(AIMessage(content=llm_response))
    
    
if __name__ == "__main__":
    main()
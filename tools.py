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


def task_manager(state: MessagesState, config: RunnableConfig):
    """Main task manager function that processes user messages and decides on actions."""
    
    # Get store from config - check both places it might be
    store = config.get("store") or config.get("configurable", {}).get("store")
    if not store:
        # Print the structure of config for debugging
        print(f"Config keys: {list(config.keys())}")
        print(f"Configurable keys: {list(config.get('configurable', {}).keys())}")
        raise ValueError("Store not found in config")
    
    # Get user ID from config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    todo_category = configurable.todo_category
    task_manager_role = configurable.task_manager_role

    # Check for task list query
    last_message = state['messages'][-1]
    last_message_content = last_message.content if hasattr(last_message, 'content') else ""
    if isinstance(last_message_content, str) and any(query in last_message_content.lower() for query in ["my tasks", "my to do", "todo list", "what are my", "what do i have", "show me my"]):
        namespace = ('todo', todo_category, user_id)
        tasks = store.search(namespace)
        
        if tasks:
            task_list = "\n".join([f"• {task.value.get('task', 'Unknown task')} (Est: {task.value.get('time_to_complete', '?')} mins)" 
                                 for task in tasks])
            response = AIMessage(content=f"Here's your current task list:\n\n{task_list}")
            return {'messages': [response]}
        else:
            response = AIMessage(content="You don't have any tasks yet. Would you like to add some?")
            return {'messages': [response]}

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
    
    # Get store from config - check both places it might be
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
    
    # Get store from config - check both places it might be
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
    
    # If it looks like a task (not a question or simple reply)
    if isinstance(content, str) and not content.lower() in ["yes", "no", "ok", "sure", "thanks", "thank you"] and not content.lower().startswith(("what", "where", "when", "how", "why", "can you", "could you")) and "?" not in content:
        # Create a new simple task
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
    
    # Get store from config - check both places it might be
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
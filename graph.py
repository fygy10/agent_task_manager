# from langgraph.graph import StateGraph, MessagesState, START

# from tools import task_manager, update_todos, update_profile, update_instructions, route_message


# #handle various configs as needed
# builder = StateGraph(MessagesState)


# #flow of nodes for memory extraction process
# builder.add_node("task_manager", task_manager)
# builder.add_node("update_todos", update_todos)
# builder.add_node("update_profile", update_profile)
# builder.add_node("update_instructions", update_instructions)


# #set up the edges
# builder.add_edge(START, 'task_manager')
# builder.add_conditional_edges('task_manager', route_message)
# builder.add_edge('update_todos', 'task_manager')
# builder.add_edge('update_profile', 'task_manager')
# builder.add_edge('update_instructions', 'task_manager')


# #compile the graph
# graph = builder.compile()

from langgraph.graph import StateGraph, MessagesState, START

# Import local dependencies
from tools import task_manager, update_todos, update_profile, update_instructions, route_message
from configuration import Configuration


def create_graph():
    """Create and return the task manager graph."""
    
    # Create the graph
    builder = StateGraph(MessagesState)
    
    # Define the flow of the memory extraction process
    builder.add_node("task_manager", task_manager)
    builder.add_node("update_todos", update_todos)
    builder.add_node("update_profile", update_profile)
    builder.add_node("update_instructions", update_instructions)
    
    # Set up the edges
    builder.add_edge(START, 'task_manager')
    builder.add_conditional_edges('task_manager', route_message)
    builder.add_edge('update_todos', 'task_manager')
    builder.add_edge('update_profile', 'task_manager')
    builder.add_edge('update_instructions', 'task_manager')
    
    # Compile the graph
    return builder.compile()


# Export a compiled instance of the graph for direct import
graph = create_graph()


# Function to run the graph with proper configuration
def run_graph(messages, user_id, todo_category, task_manager_role, store):
    """
    Run the task manager graph with the given configuration.
    
    Args:
        messages: List of messages in the conversation
        user_id: User identifier
        todo_category: Category for tasks
        task_manager_role: Description of the task manager role
        store: Memory store instance
        
    Returns:
        The result from the graph execution
    """
    config = {
        "configurable": {
            "user_id": user_id,
            "todo_category": todo_category,
            "task_manager_role": task_manager_role
        },
        "store": store
    }
    
    return graph.invoke({"messages": messages}, config)
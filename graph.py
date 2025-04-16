from langgraph.graph import StateGraph, MessagesState, START

# Import local dependencies
from tools import task_manager, update_todos, update_profile, update_instructions, route_message


# Create the graph without configuration schema
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
graph = builder.compile()
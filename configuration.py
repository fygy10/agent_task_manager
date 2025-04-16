from pydantic import BaseModel, Field
from typing import Optional, Any


class Configuration(BaseModel):
    """Configuration for the task manager application."""
    
    user_id: str = Field(description="Unique identifier for the user")
    todo_category: str = Field(description="Category for the todo items", default="general")
    task_manager_role: str = Field(
        description="Role description for the task manager",
        default="You are a helpful assistant that helps users organize their tasks and to-do lists."
    )
    store: Optional[Any] = None  # Added store field for direct access
    
    @classmethod
    def from_runnable_config(cls, config):
        """Extract configuration from a LangChain runnable config."""
        if not config or not config.get("configurable"):
            return cls(user_id="default_user")
        
        configurable = config["configurable"]
        
        # Also extract store if it exists in the config
        store = config.get("store", None)
        
        return cls(
            user_id=configurable.get("user_id", "default_user"),
            todo_category=configurable.get("todo_category", "general"),
            task_manager_role=configurable.get("task_manager_role", 
                                      "You are a helpful assistant that helps users organize their tasks and to-do lists."),
            store=store  # Pass the store to the configuration
        )
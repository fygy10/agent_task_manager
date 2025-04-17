import uuid
from langchain_core.messages import AIMessage, SystemMessage
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import re
import inspect


#manage document updates - document, intended chnages, modifications to be applied
class PatchDoc:
    
    def __init__(self, json_doc_id: str, planned_edits: str, patches: List[Dict[str, Any]]):
        self.json_doc_id = json_doc_id
        self.planned_edits = planned_edits
        self.patches = patches


#extract info from conversations
class Extractor:
    
    #llm used, tool data structure, tool to be used, indsert into document, callback function
    def __init__(self, model, tools, tool_choice=None, enable_inserts=False):
        self.model = model
        self.tools = tools
        self.tool_choice = tool_choice
        self.enable_inserts = enable_inserts
        self.listeners = []
    

    #handle llm respone with spy class
    def with_listeners(self, on_end=None):

        if on_end:
            self.listeners.append(on_end)
        return self
    

    #core extraction functionality
    def invoke(self, inputs):

        #extract from conversation and existing docs
        messages = inputs.get('messages', [])
        existing = inputs.get('existing', [])
        

        #find target tool class
        tool_class = None
        for tool in self.tools:
            if tool.__name__ == self.tool_choice:
                tool_class = tool
                break
        
        #catch if not in tools
        if not tool_class:
            raise ValueError(f"Tool {self.tool_choice} not found in available tools")
        

        #process existing documents into a dictionary for quick reference
        existing_docs = {}
        if existing:
            for key, tool_name, value in existing:
                existing_docs[key] = {'tool': tool_name, 'value': value}
        

        #schema format for the tool class - map field names to type hints
        #llm knows what to extract and how to format it
        schema = {}
        if hasattr(tool_class, '__annotations__'):
            schema = tool_class.__annotations__
        

        #system message instructing the LLM how to extract information
        schema_desc = "\n".join([f"{field}: {type_hint}" for field, type_hint in schema.items()])
        
        extraction_prompt = f"""
        You are an AI assistant specialized in extracting structured information from conversations.
        
        Your task is to analyze the conversation and extract information relevant to a {self.tool_choice} document.
        
        Here is the schema for {self.tool_choice}:
        {schema_desc}
        
        Here are some existing documents you can reference:
        {json.dumps(existing_docs, indent=2)}
        
        Instructions:
        1. Analyze the conversation for information that fits the {self.tool_choice} schema
        2. If you find relevant information, format it as a JSON object matching the schema
        3. If you don't find relevant information, respond with an empty JSON object {{}}
        4. For lists, include only unique values
        5. For free text fields, extract the most specific information possible
        
        Respond ONLY with valid JSON. Do not include any text before or after the JSON.
        """
        

        #hold messages from llm call in list
        formatted_messages = []
        formatted_messages.append(SystemMessage(content=extraction_prompt))
        
        #add conversation history to the list
        for message in messages:
            if hasattr(message, 'content') and message.content:
                if hasattr(message, 'role'):
                    formatted_messages.append({"role": message.role, "content": message.content})
                else:
                    # Assume it's a user message if no role is specified
                    formatted_messages.append({"role": "user", "content": message.content})
        
        #invoke llm to extract structured information
        llm_response = self.model.invoke(formatted_messages)
        

        #process the LLM's response
        responses = []
        response_metadata = []
        

        try:
            #get the content from the response
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            #extract JSON from the response
            json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
            json_matches = re.findall(json_pattern, content)
            
            #process json
            for json_str in json_matches:
                try:
        
                    extracted_data = json.loads(json_str)
                    
                    #skip empty objects
                    if not extracted_data:
                        continue
                    
                    #create a new instance of the tool class with the extracted data
                    doc_id = str(uuid.uuid4())
                    
                    #handle special cases for certain fields
                    if self.tool_choice == "Profile" and "interests" in extracted_data and isinstance(extracted_data["interests"], str):
                        #convert comma-separated interests to a list
                        extracted_data["interests"] = [interest.strip() for interest in extracted_data["interests"].split(",")]
                    
                    if self.tool_choice == "ToDo" and "solutions" in extracted_data and isinstance(extracted_data["solutions"], str):
                        #convert comma-separated solutions to a list
                        extracted_data["solutions"] = [solution.strip() for solution in extracted_data["solutions"].split(",")]
                    
                    #instantiate the tool class with the extracted data
                    tool_instance = tool_class(**extracted_data)
                    responses.append(tool_instance)
                    response_metadata.append({"json_doc_id": doc_id})
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error creating {self.tool_choice} instance: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error processing LLM response: {str(e)}")
        
        #call listeners with the actual LLM response
        for listener in self.listeners:
            listener(llm_response)
        
        #save the response and its extracted info to the memory state
        return {
            "responses": responses,
            "response_metadata": response_metadata
        }

# class MockRun:
#     """Mock run object for the listeners."""
    
#     def __init__(self):
#         self.child_runs = []
#         self.run_type = "chat_model"
#         self.outputs = {
#             "generations": [[{
#                 "message": {
#                     "kwargs": {
#                         "tool_calls": [
#                             {
#                                 "name": "PatchDoc",
#                                 "args": {
#                                     "json_doc_id": str(uuid.uuid4()),
#                                     "planned_edits": "Update user profile",
#                                     "patches": [
#                                         {
#                                             "value": "Updated profile information"
#                                         }
#                                     ]
#                                 }
#                             }
#                         ]
#                     }
#                 }
#             }]]
#         }


#wrapper for extractor class
def create_extractor(model, tools=None, tool_choice=None, enable_inserts=False):

    """
    
    Args:
        model: The language model to use for extraction
        tools: List of tool classes (default: None)
        tool_choice: Which tool to extract (default: None)
        enable_inserts: Whether to enable document insertions (default: False)
    
    Returns:
        An Extractor instance
    """

    #ensure empty param to avoid None error
    if tools is None:
        tools = []

    return Extractor(model, tools, tool_choice, enable_inserts)
from pydantic import BaseModel, Field
from trustcall import create_extractor


class Memory(BaseModel):
    content: str = Field(description = 'The main content of the memory. For example: User expressed interest in learning about French')


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description = 'A list of memories about the user')



#inspect the trustcall tool calls
class Spy:

    def __init__(self):
        self.called_tools = []


    def __call__(self, run):

        q = [run]
        while q:

            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == 'chat_model':
                self.called_tools.append(
                    r.outputs['generations'][0][0]['message']['kwargs']['tool_calls']

                )


#process info captured by Spy
def extract_tool_info(tool_calls, schema_name='Memory'):

    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                # Check if there are any patches
                if call['args']['patches']:
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'value': call['args']['patches'][0]['value']
                    })
                else:
                    #handle case where no changes were needed
                    changes.append({
                        'type': 'no_update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits']
                    })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    #format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        elif change['type'] == 'no_update':
            result_parts.append(
                f"Document {change['doc_id']} unchanged:\n"
                f"{change['planned_edits']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts)


    
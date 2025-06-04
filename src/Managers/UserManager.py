from pydantic import BaseModel


class ToolContext(BaseModel):
    type: str                   # none, confirmation 
    requester: str
    request: str


class UserManager:

    def __init__(self):
        self.context = ToolContext(type="none", requester="", request="")
        self.prompt: str = ""
        self.tools: list = [] # handle_IoT, handle_automation, handle_preference, handle_reverse 
        self.messages: list[dict[str,str]] = []


    def handle_user_input(self, query: str):


        if self.context.type == "confirmation":

            print("Yes confirmation context response")

            # prompt UM-LLM to classify for confirmation

            # return 

        else:

            print("No context response")
            # get UM-LLM response

            # if LLM calls tools

                # if handle_IoT

                    # get AM.IoT_call response 
                    # if successfull API call
                        # return "tell the user success"

                    # else if context request
                        # set context to confirmation
                        # return "ask the user for confirmation of..." 
                        
                    # else if information request
                        # set context to information
                        # return "ask the user for information for ... with arguments ..."
        
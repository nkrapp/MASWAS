import torch
import os
os.environ['HF_HOME'] = '/opt/local/models'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_TOKEN_PATH'] = '/home/krapp/_documents/HF_TOKEN.txt'
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers.models.llama.tokenization_llama import LlamaTokenizer
# from transformers.models.mistral.
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from typing import Optional, List, Mapping, Any
import numexpr as ne
from pydantic import BaseModel, Field

model_name = "mistralai/Mistral-Nemo-Instruct-2407"

bnb_config4 = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",           # Use fp16 for computation
    bnb_4bit_use_double_quant=True,             # Enable double quantization for better accuracy
    bnb_4bit_quant_type="nf4",                  # Use NormalFloat4 for optimal precision

)

bnb_config8 = BitsAndBytesConfig(
    load_in_8bit=True,                          # Enable 8-bit quantization
    llm_int8_enable_fp32_cpu_offload=False,     # Default, offloading to CPU not used
)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config8, device_map="cuda", )
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CalculatorArgs(BaseModel):
    """Do basic math calculations."""
    expression: str = Field(..., title = "expression", description = "numexpr of the calculation")

class Calculator(BaseTool):
    name: str = "calculator"
    description: str = "Use this tool for math operations. It requires numexpr syntax. Use it always you need to solve any math operation. Be sure syntax is correct."
    args_schema = CalculatorArgs

    def _run(self, expression: str):
        try:
            return ne.evaluate(expression).item()
        except Exception:
            return "This is not a numexpr valid syntax. Try a different syntax."

    def _arun(self, radius: int):
        raise "This tool does not support async"

class LightControllerArgs(BaseModel):
    """Control the brightness of the lights."""
    brightness: float = Field(..., description = "float between 0 and 1 to set the brightness of the lights to.")

class LightController(BaseTool):
    name: str = "Light Controller"
    description: str = "Use this tool to set the brightness of a light. As input give the 'brightness' variable in float from 0-1"
    args_schema = LightControllerArgs
    
    def _run(self, brightness: float):

        if brightness >= 0 and brightness <= 1:
            return "Setting Light to %.2f" % brightness
        else:
            return "Something went wrong, make sure to pass a float between 0 and 1"

    def _arun(self, brightness: float):
        raise "This tool does not support async"

class AirConditioningController(BaseTool):
    name: str = "Air Conditioning Controller"
    description: str = "Use this tool to set the wished temperature (in degrees Celsius) through an AC. As input give the 'temperature' variable in float between 15 and 25 degrees."

    def _run(self, temperature: float):

        if temperature >= 15 and temperature <= 25:
            return "Setting temperature to %.2f" % temperature
        else:
            return "Something went wrong, make sure to pass a float between 15 and 25"

    def _arun(self, temperature: float):
        raise "This tool does not support async"
         

class CustomLLMMistral(LLM):
    model: MistralForCausalLM
    tokenizer: PreTrainedTokenizerFast

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:

        messages = [
            {"role": "user", "content": prompt},
        ]

        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id, top_k=4, temperature=0.7)
        decoded = self.tokenizer.batch_decode(generated_ids)

        output = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

        if stop is not None:
          for word in stop:
            output = output.split(word)[0].strip()

        while not output.endswith("```"):
          output += "`"

        return output


calculator_tool = Calculator()
light_switch_tool = LightController()
AC_tool = AirConditioningController()
tools = [calculator_tool, light_switch_tool, AC_tool]

system="""
You are designed to solve tasks to control a smart workspace. Each task requires multiple steps that are represented by a markdown code snippet of a json blob.
The json structure should contain the following keys:
thought -> your thoughts
action -> name of a tool
action_input -> parameters to send to the tool

These are the tools you can use: {tool_names}.

These are the tools descriptions:

{tools}

If you have enough information to answer the query use the tool "Final Answer". Its parameters is the solution.
If there is not enough information, keep trying.

"""
human="""
Add the word "STOP" after each markdown snippet. Example:

```json
{{"thought": "<your thoughts>",
 "action": "<tool name or Final Answer to give a final answer>",
 "action_input": "<tool parameters or the final output"}}
```
STOP

This is my query="{input}". Write only the next step needed to solve it.
Your answer should be based in the previous tools executions, even if you think you know the answer.
Remember to add STOP after each snippet.

These were the previous steps given to solve this query and the information you already gathered:
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", human),
    MessagesPlaceholder("agent_scratchpad")  
])

llm = CustomLLMMistral(model=model, tokenizer=tokenizer)

agent = create_json_chat_agent(
    tools = tools,
    llm = llm,
    prompt = prompt,
    stop_sequence = ["STOP"],
    template_tool_response = "{observation}"
)

agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    verbose = True,
    handle_parsing_errors = True
)

agent_executor.invoke({"input": "How much is 23 plus 17?"})
agent_executor.invoke({"input": "please dim the light a little bit"})
agent_executor.invoke({"input": "Turn the temperature to 20 degrees please"})
agent_executor.invoke({"input": "Turn the temperature to 292 kelvin please"})

# print(f"The gerate test resulted in: \n{generate_test}")
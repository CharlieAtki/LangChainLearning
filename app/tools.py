from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.tools import tool
from langchain_core.messages import ToolMessage, AIMessage

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# -----------------------------
# Pydantic model
# -----------------------------
class Multiplier(BaseModel):
    factor: int
    number: int
    result: int


# -----------------------------
# TOOL
# -----------------------------
@tool("multiplier")
def multiplier(factor: int, number: int) -> int:
    """Multiplies a number by a given factor."""
    return factor * number


# Map for lookup
tool_map = {"multiplier": multiplier}


# -----------------------------
# Parser
# -----------------------------
parser = PydanticOutputParser(pydantic_object=Multiplier)

\
# -----------------------------
# First prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant. Never do maths yourself. "
        "Always call the tool instead."
    ),
    ("human", "Factor: {factor}, Number: {number}")
])


# -----------------------------
# LLM with tools bound
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")  # Fixed: gpt-4o-mini instead of gpt-5-mini
llm_with_tools = llm.bind_tools([multiplier])


# -----------------------------
# TOOL EXECUTOR LAYER (IMPORTANT!)
# -----------------------------
def execute_tool(response: AIMessage):
    # Check if there are tool calls
    if not response.tool_calls:
        return response  # nothing to do

    # Get the first tool call
    tool_call = response.tool_calls[0]
    
    # Get the tool function
    tool_fn = tool_map[tool_call["name"]]
    
    # Execute the tool
    result = tool_fn.invoke(tool_call["args"])

    # Return a ToolMessage
    return ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )

tool_executor = RunnableLambda(execute_tool)


# -----------------------------
# Second prompt (LLM uses tool result)
# -----------------------------
final_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "Use the tool result to complete the schema. "
     "Output valid JSON matching the Multiplier schema: "
     "{format_instructions}"),
    ("human", "Tool result: {tool_result}")
])


# -----------------------------
# Format the final result for parsing
# -----------------------------
def format_for_parser(tool_message: ToolMessage):
    # We need to pass the result back to the LLM to format it properly
    return final_prompt.format(
        format_instructions=parser.get_format_instructions(),
        tool_result=tool_message.content
    )


# -----------------------------
# LCEL chain
# -----------------------------
multiplication_chain = (
    prompt
    | llm_with_tools
    | tool_executor
    | RunnableLambda(format_for_parser)
    | llm
    | parser
)

# -----------------------------
# RUN IT
# -----------------------------
result = multiplication_chain.invoke({"factor": 5, "number": 10})
print(result)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ActionItem(BaseModel):
    industry: str
    businessIdea: str
    targetMarket: str
    weaknesses: str
    strengths: str

# Takes LLM text and parses it into a structured Pydantic model
parser = PydanticOutputParser(pydantic_object=ActionItem)

prompt = ChatPromptTemplate.from_messages([ # Defines a chat-style prompt with system + human roles
    (
        "system",
        "You are an AI Business Advisor. Generate a business concept report "
        "based on the industry provided.\n\n"
        "Format your final answer using this schema:\n{format_instructions}" # Filled in with the parsers instructions, so the model knows how to output JSON
    ),
    ("human", "Industry: {industry}")
]).partial(format_instructions=parser.get_format_instructions()) # .partial() pre-fills in the format_instructions variable

llm = ChatOpenAI(model="gpt-5-mini")



chain = (
    {"industry": RunnablePassthrough()}
    | prompt # Injects the input into the prompt template
    | llm # calls the model with the prompt
    | parser # Converts the output into a Pydantic ActionItem object
)

# Sends "Technology as the industry input"
# Output ActionItem object
result = chain.invoke("Technology")

print(result)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Pydantic models defining the structured outputs
class BusinessIdea(BaseModel):
    industry: str
    businessIdea: str
    targetMarket: str

class AnalysisReport(BaseModel):
    summary: str
    recommendations: str



# Takes LLM text and parses it into a structured Pydantic model
idea_parser = PydanticOutputParser(pydantic_object=BusinessIdea)

idea_prompt = ChatPromptTemplate.from_messages([ # Defines a chat-style prompt with system + human roles
    (
        "system",
        "You are an AI Business Advisor. Generate a business concept report "
        "based on the industry provided.\n\n"
        "Format your final answer using this schema:\n{format_instructions}" # Filled in with the parsers instructions, so the model knows how to output JSON
    ),
    ("human", "Industry: {industry}")
]).partial(format_instructions=idea_parser.get_format_instructions()) # .partial() pre-fills in the format_instructions variable

llm = ChatOpenAI(model="gpt-5-mini")



idea_chain = (
    {"industry": RunnablePassthrough()}
    | idea_prompt # Injects the input into the prompt template
    | llm # calls the model with the prompt
    | idea_parser # Converts the output into a Pydantic ActionItem object
)

# Sends "Technology as the industry input"
# Output ActionItem object
try:
    idea_result = idea_chain.invoke("Technology")
except Exception as e:
    print("Error during idea generation:", e)
    idea_result = None

analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert business analyst. Analyze the following business idea and provide a summary and recommendations.\n\n"
        "Format your final answer using this schema:\n{format_instructions}"
    ),
    (
        "human",
        "Business Idea:\n"
        "Industry: {industry}\n"
        "Business Idea: {businessIdea}\n"
        "Target Market: {targetMarket}\n"
    )
]).partial(format_instructions=PydanticOutputParser(pydantic_object=AnalysisReport).get_format_instructions())


analysis_chain = (
    {"industry": RunnablePassthrough(), "businessIdea": RunnablePassthrough(), "targetMarket": RunnablePassthrough()}
    | analysis_prompt
    | llm
    | PydanticOutputParser(pydantic_object=AnalysisReport)
)

try:
    analysis_result = analysis_chain.invoke({
        "industry": idea_result.industry,
        "businessIdea": idea_result.businessIdea,
        "targetMarket": idea_result.targetMarket
    })
except Exception as e:
    print("Error during analysis:", e)
    analysis_result = None

print("\nBusiness Idea:")
for field, value in idea_result.model_dump().items():
    print(f"{field}: {value}\n")

print("\nAnalysis Report:")
for field, value in analysis_result.model_dump().items():
    print(f"{field}: {value}\n")

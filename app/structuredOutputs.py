from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# Structured output model definition -> Defines how the LLM should format its response
class TripPlan(BaseModel):
    title: str
    location: str
    date: str

llm = ChatOpenAI(model="gpt-5-mini")

history = [
    SystemMessage("You are a travel agent helping users plan trips.")
]

def ask(question):
    history.append(HumanMessage(question))

    # Structured output model
    response = llm.with_structured_output(TripPlan).invoke(history)

    # Convert the TripPlan model to a string before storing as a message
    history.append(AIMessage(response.model_dump_json()))

    print("\nAssistant:", response.content, "\n")

ask("I would like to plan a trip to Paris. Can you help?") # Initial question to start planning a trip
ask("I am going on the 14th August") # Follow-up question to test context retention 


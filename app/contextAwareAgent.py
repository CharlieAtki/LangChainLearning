from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()
 
llm = ChatOpenAI(
    model="gpt-5-mini",
)

# Conversation history to retain context
history = [
    SystemMessage("You are a geography tutor"),
]

# Function to handle conversation with context retention
def ask(question):
    history.append(HumanMessage(question)) # Append the user's question to history
    response = llm.invoke(history) # Invoke the model with the conversation history
    history.append(AIMessage(response.content)) # Append the agents response to history
    print("\nAssistant:", response.content, "\n")

# interactions -> Conversation with context retention
ask("What is the capital of France?")
ask("Where was the capital again?")


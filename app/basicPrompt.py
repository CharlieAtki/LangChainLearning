from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()
 
llm = ChatOpenAI(
    model="gpt-5-mini",
)

history = [
    SystemMessage("You are a geography tutor"),
]

def ask(question):
    history.append(HumanMessage(question))
    response = llm.invoke(history)
    history.append(AIMessage(response.content))
    print("\nAssistant:", response.content, "\n")

ask("What is the capital of France?")
ask("Where was the capital again?")

# prompt_template = PromptTemplate(template="Tell me a joke about {topic}")
# llm.invoke(prompt_template.format(topic="Java"))

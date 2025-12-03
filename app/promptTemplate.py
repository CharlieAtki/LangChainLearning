from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Defining the language model
llm = ChatOpenAI(
    model="gpt-5-mini",
    )

# Creating a prompt template for few-shot learning
example_prompt = PromptTemplate(
    template="Question: {input}\nThought: {thought}\nResponse: {output}",
)

# Giving the model few-shot examples to improve its performance
examples = [
    {
        "input": "If a store applies a 20% discount to a $50 item, what is the sale price?",
        "thought": "A 20% discount means multiplying the original price by 0.8. So, $50 * 0.8 = $40.",
        "output": "$40"
    }
]

# Creating the few-shot prompt template
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

# Invoking the model with the few-shot prompt template
response = llm.invoke(
    prompt_template.invoke({"input":"If I apply a 50% discount to a $80 item, what is the sale price?"})
)

print("\n" + response.content + "\n")

# Alternative approach using messages directly ->
# messages = [
#     SystemMessage("You are a geography tutor"),
#     HumanMessage("What is the capital of France?"),
#     AIMessage("The capital of France is Paris."),
#     HumanMessage("What is the capital of Germany?")
# ]

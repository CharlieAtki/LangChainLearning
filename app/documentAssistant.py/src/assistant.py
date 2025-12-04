from schemas import AgentState, UserIntent, AnswerResponse
from prompts import get_intent_classification_prompt

def triage_agent_node(state: AgentState, config) -> AgentState: # Agent to classify user intent
    llm = config["llm"].with_structured_output(UserIntent)

    # Fetches the prompt template and fills in the variables
    # .format() replaces {user_input} and {conversation_history} with the actual values
    prompt = get_intent_classification_prompt().format(
        user_input=state.user_input,
        conversation_history=state.messages
    )

    intent: UserIntent = llm.invoke(prompt)

    state.intent = intent

    # Validate intent and route to appropriate next step
    state.next_step = (
        "qa_agent" if intent.intent_type == "qa" else
        "summarization_agent" if intent.intent_type == "summarization" else
        "calculation_agent" if intent.intent_type == "calculation" else
        "qa_agent"
    )

    state.actions_taken.append("classify_intent")
    return state

def qa_agent_node(state: AnswerResponse) -> AnswerResponse:
    response = llm.invoke(prompt)
    answer = response.content.strip()

    return AnswerResponse(
        input=state.input,
        action=state.action,
        result=answer
    )
    

def summarisation_agent_node(state: AnswerResponse) -> AnswerResponse:
    prompt = f"User request: {state.input}\nProvide a concise summary based on the documents."
    response = llm.invoke(prompt)
    summary = response.content.strip()

    return AnswerResponse(
        input=state.input,
        action=state.action,
        result=summary
    )


from schemas import AgentState, UserIntent, AnswerResponse
from prompts import get_intent_classification_prompt, get_chat_prompt_template

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

def qa_agent_node(state: AgentState, config):
    llm = config["llm"].with_structured_output(AnswerResponse)

    prompt = get_chat_prompt_template("qa").format(
        user_input=state.user_input,
        conversation_history=state.messages,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents
    )

    # LLM returns structured Pydantic AnswerResponse
    answer: AnswerResponse = llm.invoke(prompt)

    # Update agent state
    state.current_response = answer
    state.next_step = None  # or END depending on design
    state.actions_taken.append("qa_agent")

    return state

    
def summarisation_agent_node(state: AgentState, config):
    # Configure LLM with structured output
    llm = config["llm"].with_structured_output(SummaryResponse)

    # Prepare the prompt
    prompt = get_chat_prompt_template("summarization").format(
        user_input=state.user_input,
        conversation_history=state.messages,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents
    )

    # Generate the structured summary
    summary: SummaryResponse = llm.invoke(prompt)

    # Update agent state
    state.current_response = summary
    state.actions_taken.append("summarisation_agent")
    state.next_step = None  # workflow will go → update_memory

    return state

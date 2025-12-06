from app.tools import llm_with_tools
from schemas import AgentState, UserIntent, AnswerResponse
from prompts import get_intent_classification_prompt, get_chat_prompt_template
from tools import retrieve_documents, search_specific_document
from datetime import datetime

def triage_agent_node(state: AgentState, config) -> AgentState: # Agent to classify user intent
    # Access LLM from the configurable section
    llm = config["configurable"]["llm"].with_structured_output(UserIntent)

    # Fetches the prompt template and fills in the variables
    # .format() replaces {user_input} and {conversation_history} with the actual values
    prompt = get_intent_classification_prompt().format(
        user_input=state.user_input,
        conversation_history=state.messages
    )

    intent: UserIntent = llm.invoke(prompt)

    # Create a new state with updated values
    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=intent,
        next_step=(
            "qa_agent" if intent.intent_type == "qa" else
            "summarisation_agent" if intent.intent_type == "summarisation" else  # Fixed: changed from "summarization" to "summarisation"
            "calculation_agent" if intent.intent_type == "calculation" else
            "qa_agent"
        ),
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=state.current_response,
        tools_used=state.tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["classify_intent"],
        retrieved_documents=state.retrieved_documents
    )

def qa_agent_node(state: AgentState, config):
    # Use function_calling method to avoid the warning
    llm = config["configurable"]["llm"].with_structured_output(AnswerResponse, method="function_calling")
    tools = [retrieve_documents, search_specific_document]
    llm_with_tools = llm.bind_tools(tools)

    prompt = get_chat_prompt_template("qa").format(
        user_input=state.user_input,
        conversation_history=state.messages,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents
    )

    # LLM returns structured Pydantic AnswerResponse
    answer: AnswerResponse = llm_with_tools.invoke(prompt)

    # Create a new state with updated values
    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=state.intent,
        next_step=None,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=answer,
        tools_used=state.tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["qa_agent"]
    )

def summarisation_agent_node(state: AgentState, config):
    # Configure LLM with structured output using function_calling
    llm = config["configurable"]["llm"].with_structured_output(AnswerResponse, method="function_calling")

    # Prepare the prompt
    prompt = get_chat_prompt_template("summarization").format(
        user_input=state.user_input,
        conversation_history=state.messages,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents
    )

    # Generate the structured summary
    summary: AnswerResponse = llm.invoke(prompt)

    # Create a new state with updated values
    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=state.intent,
        next_step=None,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=summary,
        tools_used=state.tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["summarisation_agent"]
    )
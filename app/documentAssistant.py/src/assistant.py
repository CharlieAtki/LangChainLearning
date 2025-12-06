from schemas import AgentState, UserIntent, AnswerResponse
from prompts import get_intent_classification_prompt, get_chat_prompt_template
from tools import retrieve_documents, search_specific_document, calculate
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


def triage_agent_node(state: AgentState, config) -> AgentState:
    """Agent to classify user intent"""
    llm = config["configurable"]["llm"].with_structured_output(UserIntent)

    prompt = get_intent_classification_prompt().format(
        user_input=state.user_input,
        conversation_history=state.messages
    )

    intent: UserIntent = llm.invoke(prompt)

    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=intent,
        next_step=(
            "qa_agent" if intent.intent_type == "qa" else
            "summarisation_agent" if intent.intent_type == "summarisation" else
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
    """QA agent with tool calling capability"""
    llm = config["configurable"]["llm"]
    tools = [retrieve_documents, search_specific_document]
    llm_with_tools = llm.bind_tools(tools)

    # Get the system prompt
    system_prompt = get_chat_prompt_template("qa")

    # Create the initial message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.user_input}
    ]

    tools_used = []
    retrieved_docs = []

    # Tool calling loop - allow LLM to use tools iteratively
    max_iterations = 5
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)

        # Check if the LLM wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                # Execute the appropriate tool
                if tool_name == "retrieve_documents":
                    tool_result = retrieve_documents.invoke(tool_args)
                    retrieved_docs.extend(tool_result)
                elif tool_name == "search_specific_document":
                    tool_result = search_specific_document.invoke(tool_args)
                    retrieved_docs.append(tool_result)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}

                tools_used.append(tool_name)

                # Add tool response to messages
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.get('id', 'unknown')
                })
        else:
            # No more tool calls - we have the final answer
            final_answer = response.content if hasattr(response, 'content') else str(response)

            # Extract sources from retrieved documents
            sources = list(set([doc.get('id', doc.get('document_id', 'unknown'))
                                for doc in retrieved_docs]))

            # Create structured response
            answer_response = AnswerResponse(
                question=state.user_input,
                answer=final_answer,
                sources=sources,
                confidence=0.85,
                timestamp=datetime.now(),
                retrieved_documents=retrieved_docs
            )

            return AgentState(
                user_input=state.user_input,
                messages=state.messages + [state.user_input, final_answer],
                intent=state.intent,
                next_step=None,
                conversation_summary=state.conversation_summary,
                active_documents=state.active_documents,
                current_response=answer_response,
                tools_used=state.tools_used + tools_used,
                session_id=state.session_id,
                user_id=state.user_id,
                actions_taken=state.actions_taken + ["qa_agent"],
                retrieved_documents=state.retrieved_documents
            )

    # If we hit max iterations, return what we have
    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=state.intent,
        next_step=None,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=AnswerResponse(
            question=state.user_input,
            answer="I couldn't complete the task within the allowed iterations.",
            sources=[],
            confidence=0.0,
            timestamp=datetime.now()
        ),
        tools_used=state.tools_used + tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["qa_agent"],
        retrieved_documents=state.retrieved_documents
    )


def summarisation_agent_node(state: AgentState, config):
    """Summarization agent with tool calling capability"""
    llm = config["configurable"]["llm"]
    tools = [retrieve_documents, search_specific_document]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = get_chat_prompt_template("summarization")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.user_input}
    ]

    tools_used = []
    retrieved_docs = []

    # Tool calling loop
    max_iterations = 5
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                if tool_name == "retrieve_documents":
                    tool_result = retrieve_documents.invoke(tool_args)
                    retrieved_docs.extend(tool_result)
                elif tool_name == "search_specific_document":
                    tool_result = search_specific_document.invoke(tool_args)
                    retrieved_docs.append(tool_result)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}

                tools_used.append(tool_name)

                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.get('id', 'unknown')
                })
        else:
            final_summary = response.content if hasattr(response, 'content') else str(response)

            sources = list(set([doc.get('id', doc.get('document_id', 'unknown'))
                                for doc in retrieved_docs]))

            summary_response = AnswerResponse(
                question=state.user_input,
                answer=final_summary,
                sources=sources,
                confidence=0.85,
                timestamp=datetime.now(),
                retrieved_documents=retrieved_docs
            )

            return AgentState(
                user_input=state.user_input,
                messages=state.messages + [state.user_input, final_summary],
                intent=state.intent,
                next_step=None,
                conversation_summary=state.conversation_summary,
                active_documents=state.active_documents,
                current_response=summary_response,
                tools_used=state.tools_used + tools_used,
                session_id=state.session_id,
                user_id=state.user_id,
                actions_taken=state.actions_taken + ["summarisation_agent"],
                retrieved_documents=state.retrieved_documents
            )

    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=state.intent,
        next_step=None,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=AnswerResponse(
            question=state.user_input,
            answer="I couldn't complete the summarization within the allowed iterations.",
            sources=[],
            confidence=0.0,
            timestamp=datetime.now()
        ),
        tools_used=state.tools_used + tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["summarisation_agent"],
        retrieved_documents=state.retrieved_documents
    )


def calculation_agent_node(state: AgentState, config):
    """Calculation agent with tool calling capability"""
    llm = config["configurable"]["llm"]
    tools = [retrieve_documents, search_specific_document, calculate]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = get_chat_prompt_template("calculation")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state.user_input}
    ]

    tools_used = []
    retrieved_docs = []

    max_iterations = 5
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)

        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                if tool_name == "retrieve_documents":
                    tool_result = retrieve_documents.invoke(tool_args)
                    retrieved_docs.extend(tool_result)
                elif tool_name == "search_specific_document":
                    tool_result = search_specific_document.invoke(tool_args)
                    retrieved_docs.append(tool_result)
                elif tool_name == "calculate":
                    tool_result = calculate.invoke(tool_args)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}

                tools_used.append(tool_name)

                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.get('id', 'unknown')
                })
        else:
            final_answer = response.content if hasattr(response, 'content') else str(response)

            sources = list(set([doc.get('id', doc.get('document_id', 'unknown'))
                                for doc in retrieved_docs]))

            calc_response = AnswerResponse(
                question=state.user_input,
                answer=final_answer,
                sources=sources,
                confidence=0.9,
                timestamp=datetime.now(),
                retrieved_documents=retrieved_docs
            )

            return AgentState(
                user_input=state.user_input,
                messages=state.messages + [state.user_input, final_answer],
                intent=state.intent,
                next_step=None,
                conversation_summary=state.conversation_summary,
                active_documents=state.active_documents,
                current_response=calc_response,
                tools_used=state.tools_used + tools_used,
                session_id=state.session_id,
                user_id=state.user_id,
                actions_taken=state.actions_taken + ["calculation_agent"],
                retrieved_documents=state.retrieved_documents
            )

    return AgentState(
        user_input=state.user_input,
        messages=state.messages,
        intent=state.intent,
        next_step=None,
        conversation_summary=state.conversation_summary,
        active_documents=state.active_documents,
        current_response=AnswerResponse(
            question=state.user_input,
            answer="I couldn't complete the calculation within the allowed iterations.",
            sources=[],
            confidence=0.0,
            timestamp=datetime.now()
        ),
        tools_used=state.tools_used + tools_used,
        session_id=state.session_id,
        user_id=state.user_id,
        actions_taken=state.actions_taken + ["calculation_agent"],
        retrieved_documents=state.retrieved_documents
    )
from schemas import UserIntent, AnswerResponse

def triage_agent_node(state: UserIntent) -> UserIntent: # Agent to classify user intent
    prompt = f"User input: {state.input}\nDecide whether this is a 'question', 'summarise', or 'calculation'. Only output one word. If none apply, say 'none'."
    response = llm.invoke(prompt)
    action = response.content.strip().lower()

    if action not in ["question", "summarise", "calculation"]:
        action = None

    return UserIntent(input=state.input, action=action)

def qa_agent_node(state: AnswerResponse) -> AnswerResponse:
    prompt = f"User question: {state.input}\nProvide a concise answer based on the documents."
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

def calculation_agent_node(state: AnswerResponse) -> AnswerResponse:

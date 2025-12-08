from langchain_openai import ChatOpenAI
from agent import agent_workflow
from schemas import AgentState
from tools import retrieve_documents, search_specific_document, calculate
from dotenv import load_dotenv
import uuid

load_dotenv()

def main():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-5-mini")
    
    # Create a session ID for this conversation
    session_id = str(uuid.uuid4())
    
    # Create configuration that will be passed to agents
    # This includes the thread_id for checkpointer persistence
    config = {
        "configurable": {
            "thread_id": session_id,  # Required for checkpointer
            "llm": llm,
            "tools": [retrieve_documents, search_specific_document, calculate]
        }
    }
    
    # CRITICAL: Create the workflow ONCE and reuse it
    # Each call to agent_workflow() creates a NEW checkpointer
    workflow = agent_workflow()
    
    # Create initial state using the Pydantic model
    # Note: Don't initialize actions_taken=[] when using checkpointer
    # The reducer will handle it properly
    initial_state = AgentState(
        user_input="Can you summarise doc_1?",
        messages=[],  # Start with empty conversation history
        conversation_summary="",
        active_documents=[],  # Your active documents if any
        next_step=None,
        intent=None,
        current_response=None,
        session_id=session_id
    )
    
    # Run the workflow - pass state as input and config separately
    # With checkpointer, state will be persisted
    result = workflow.invoke(initial_state, config)
    
    # Process the result - result is dict but nested objects are still Pydantic models
    if result.get("current_response"):
        print(f"Response: {result['current_response'].answer}")  # Use dot notation for Pydantic model
        print(f"Actions taken: {result['actions_taken']}")
    else:
        print("No response generated")
        print(f"Actions taken: {result.get('actions_taken', [])}")
        
    # Also print the intent for debugging
    if result.get("intent"):
        print(f"Detected intent: {result['intent'].intent_type} (confidence: {result['intent'].confidence})")
    
    print(f"\nSession ID: {session_id}")
    print("State has been persisted. You can continue this conversation by using the same thread_id.")
    
    # Example: Run another query in the same session
    print("\n" + "="*50)
    print("Running second query in same session...")
    print("="*50 + "\n")
    
    second_state = AgentState(
        user_input="What was the last question?",
        session_id=session_id
    )
    
    # CRITICAL: Use the SAME workflow instance to access the same checkpointer
    result2 = workflow.invoke(second_state, config)
    
    if result2.get("current_response"):
        print(f"Response: {result2['current_response'].answer}")
        print(f"Actions taken (accumulated): {result2['actions_taken']}")
    
    print(f"\nTotal actions across both invocations: {result2.get('actions_taken', [])}")

if __name__ == "__main__":
    main()

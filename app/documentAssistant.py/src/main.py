from langchain_openai import ChatOpenAI
from agent import agent_workflow
from schemas import AgentState

from dotenv import load_dotenv

load_dotenv()

def main():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-5-mini")
    
    # Create configuration that will be passed to agents
    config = {
        "configurable": {
            "llm": llm
        }
    }
    
    # Create initial state using the Pydantic model
    initial_state = AgentState(
        user_input="Can you summarise what data visualisation means?",
        messages=[],  # Start with empty conversation history
        conversation_summary="",
        active_documents=[],  # Your active documents if any
        actions_taken=[],
        next_step=None,
        intent=None,
        current_response=None
    )
    
    # Run the workflow - pass state as input and config separately
    result = agent_workflow().invoke(initial_state, config)
    
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

if __name__ == "__main__":
    main()
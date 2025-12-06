from langchain.tools import tool
from typing import List, Dict, Any
import json

@tool
def retrieve_documents(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
        Retrieve documents based on a search query.

        Args:
            query: The search query to find relevant documents
            max_results: Maximum number of documents to return

        Returns:
            List of documents with id, content, and metadata
        """
    # This is a mock implementation - replace with your actual document retrieval logic
    # You might use vector databases like Chroma, Pinecone, or FAISS

    mock_documents = [
        {
            "id": "doc_1",
            "content": "Data visualization is the graphical representation of information and data...",
            "title": "Introduction to Data Visualization",
            "source": "data_viz_guide.pdf"
        },
        {
            "id": "doc_2",
            "content": "Effective data visualization helps in understanding complex datasets...",
            "title": "Best Practices in Data Visualization",
            "source": "viz_best_practices.pdf"
        }
    ]

    # Simple keyword matching for demo - replace with proper vector search
    relevant_docs = [doc for doc in mock_documents if
                     any(word in doc["content"].lower() for word in query.lower().split())]

    return relevant_docs[:max_results]


@tool
def search_specific_document(document_id: str, query: str) -> Dict[str, Any]:
    """
    Search for specific information within a particular document.

    Args:
        document_id: The ID of the document to search
        query: What to search for within the document

    Returns:
        Document section matching the query
    """
    # Mock implementation - replace with your document search logic
    return {
        "document_id": document_id,
        "matching_section": f"Found relevant content for '{query}' in document {document_id}",
        "confidence": 0.85
    }


@tool
def calculate(expression: str) -> str | float:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The calculated result
    """
    try:
        # Simple eval for demo - use a safer math parser in production
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

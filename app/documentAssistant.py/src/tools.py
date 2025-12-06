from langchain_core.tools import tool
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
    # Enhanced mock documents database
    mock_documents = [
        {
            "id": "doc_1",
            "content": "Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data. In the world of Big Data, data visualization tools and technologies are essential to analyze massive amounts of information and make data-driven decisions.",
            "title": "Introduction to Data Visualization",
            "source": "data_viz_guide.pdf"
        },
        {
            "id": "doc_2",
            "content": "Effective data visualization helps in understanding complex datasets by presenting information in a visual context. Key principles include choosing the right chart type, using appropriate colors, maintaining simplicity, and ensuring accessibility. Common visualization types include bar charts, line graphs, scatter plots, heat maps, and tree maps.",
            "title": "Best Practices in Data Visualization",
            "source": "viz_best_practices.pdf"
        },
        {
            "id": "doc_3",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "title": "Introduction to Machine Learning",
            "source": "ml_basics.pdf"
        },
        {
            "id": "doc_4",
            "content": "Python data visualization libraries include Matplotlib for basic plotting, Seaborn for statistical visualizations, Plotly for interactive charts, and Bokeh for web-based visualizations. Each library has its strengths and use cases depending on the project requirements.",
            "title": "Python Visualization Libraries",
            "source": "python_viz.pdf"
        },
        {
            "id": "doc_5",
            "content": "The quarterly revenue for Q1 was $1,250,000, Q2 was $1,450,000, Q3 was $1,680,000, and Q4 was $1,920,000. Total annual revenue reached $6,300,000 representing a 15% year-over-year growth.",
            "title": "2024 Financial Report",
            "source": "finance_2024.pdf"
        }
    ]

    # Improved keyword matching - case insensitive and handles partial matches
    query_lower = query.lower()
    query_words = query_lower.split()

    # Score each document based on relevance
    scored_docs = []
    for doc in mock_documents:
        score = 0
        doc_text = (doc["content"] + " " + doc["title"]).lower()

        # Check for exact phrase match (higher score)
        if query_lower in doc_text:
            score += 10

        # Check for individual word matches
        for word in query_words:
            if len(word) > 2:  # Ignore very short words
                if word in doc_text:
                    score += 1

        if score > 0:
            scored_docs.append((score, doc))

    # Sort by relevance score (descending)
    scored_docs.sort(reverse=True, key=lambda x: x[0])

    # Return top results
    relevant_docs = [doc for score, doc in scored_docs[:max_results]]

    print(f"[TOOL] retrieve_documents called with query='{query}', found {len(relevant_docs)} documents")

    return relevant_docs


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
    # Mock implementation - in production, this would search within the actual document
    print(f"[TOOL] search_specific_document called for doc_id='{document_id}', query='{query}'")

    return {
        "document_id": document_id,
        "matching_section": f"Found relevant content for '{query}' in document {document_id}. This section discusses {query} in detail with specific examples and technical explanations.",
        "confidence": 0.85
    }


@tool
def calculate(expression: str) -> float | str:
    """
    Perform mathematical calculations.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2", "10 * 5", "1250000 + 1450000 + 1680000 + 1920000")

    Returns:
        The calculated result as a float
    """
    try:
        print(f"[TOOL] calculate called with expression='{expression}'")
        # Use eval with restricted namespace for safety
        # In production, use a proper math parser like ast.literal_eval or sympy
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        print(f"[TOOL] calculate error: {str(e)}")
        return f"Error in calculation: {str(e)}"
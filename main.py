import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from typing import TypedDict
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


os.environ["SERPER_API_KEY"] = "YOUR_SERPER_API_KEY"
llm_text = OllamaLLM(model="qwen2.5:latest", num_gpu=1, max_tokens=250)
llm_vision = OllamaLLM(model="llava:13b", num_gpu=1)

class AgentState(TypedDict):
    image_path: str
    image_description: str
    search_query: str
    search_results: dict
    recommendations: list

def process_image(state: AgentState) -> AgentState:
    try:
        
        # Use Ollama vision model to describe the image
        template = """Describe the clothing item in the image in detail, including type, color, style, and any distinctive features only in one sentence.
        image: {image}
        image description:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm_vision | StrOutputParser()
        description = chain.invoke({"image": f"./{state['image_path']}"})
        
        return {
            **state,
            "image_description": description
        }
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return state

def generate_query(state: AgentState) -> AgentState:
    try:
        if "image_description" not in state or not state["image_description"]:
            raise ValueError("No image description available")
            
        template = """Generate a search query to find similar clothing items online based on this description: {description}.
        The search query must be at most three word.
        search query:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm_text | StrOutputParser()
        query = chain.invoke({"description": state["image_description"]})
        
        return {
            **state,
            "search_query": query.strip('"')
        }
    except Exception as e:
        print(f"Error generating query: {e}")
        return state

def search_internet(state: AgentState) -> AgentState:
    try:
        if "search_query" not in state or not state["search_query"]:
            raise ValueError("No search query available")
            
        search = GoogleSerperAPIWrapper()
        results = search.results(state["search_query"])
        
        return {
            **state,
            "search_results": results if isinstance(results, dict) else {"organic": []}
        }
    except Exception as e:
        print(f"Error searching internet: {e}")
        return {
            **state,
            "search_results": {"organic": []}
        }

def recommend_stores(state: AgentState) -> AgentState:
    try:
        if "search_results" not in state:
            raise ValueError("No search results available")
            
        organic_results = state["search_results"].get("organic", [])
        recommendations = [result.get("link", "") for result in organic_results[:5]]  # Limit to top 5
        
        return {
            **state,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error recommending stores: {e}")
        return {
            **state,
            "recommendations": []
        }

# Build the workflow
workflow = StateGraph(AgentState)
workflow.add_node("process_image", process_image)
workflow.add_node("generate_query", generate_query)
workflow.add_node("search_internet", search_internet)
workflow.add_node("recommend_stores", recommend_stores)

workflow.set_entry_point("process_image")
workflow.add_edge("process_image", "generate_query")
workflow.add_edge("generate_query", "search_internet")
workflow.add_edge("search_internet", "recommend_stores")
workflow.add_edge("recommend_stores", END)

# Compile the graph
graph = workflow.compile()

# Run the graph with initial state
initial_state = AgentState(
    image_path="red_skirt2.jpg",
    image_description="",
    search_query="",
    search_results={},
    recommendations=[]
)

try:
    final_state = graph.invoke(initial_state)
    print("\nRecommendations:")
    for i, rec in enumerate(final_state.get("recommendations", []), 1):
        print(f"{i}. {rec}")
except Exception as e:
    print(f"Error in workflow execution: {e}")
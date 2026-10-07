"""A small production-style agent module."""
import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

load_dotenv(find_dotenv(usecwd=True))
llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"), temperature=0, max_retries=2)


@tool
def word_count(text: str) -> int:
    """Count the words in a text."""
    return len(text.split())


tools = [word_count]
llm_with_tools = llm.bind_tools(tools)


def agent(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent, retry_policy=RetryPolicy(max_attempts=3))
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile()          # the deploy server adds its own persistence

from langgraph.graph import StateGraph, START
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict
from dotenv import load_dotenv


class State(TypedDict):
    pass


load_dotenv()
model = init_chat_model("openai:gpt-4o")


def ask_llm(_: State) -> State:
    user_query = input("query: ")
    answer_message: AIMessage = model.invoke([HumanMessage(user_query)])
    print("answer: ", answer_message.content)

    return {}


graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)

graph.add_edge(START, "ask_llm")
graph.add_edge("ask_llm", "ask_llm")

workflow = graph.compile()

with open("graph2.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

workflow.invoke({}, {"recursion_limit": 100})

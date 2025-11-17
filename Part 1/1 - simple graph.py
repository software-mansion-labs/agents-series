from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    a: int
    b: int


def get_user_data(_: State) -> State:
    a = int(input("a = "))
    b = int(input("b = "))
    return {"a": a, "b": b}


def loop_condition(_: State) -> State:
    return {}


def modify(state: State) -> State:
    a, b = state["a"], state["b"]
    a, b = b, a % b
    return {"a": a, "b": b}


def write(state: State) -> State:
    print("GCD = ", state["a"])
    return {}


graph = StateGraph(State)

graph.add_node("get_user_data", get_user_data)
graph.add_node("loop_condition", loop_condition)  # dummy node
graph.add_node("modify", modify)
graph.add_node("write", write)

graph.add_edge(START, "get_user_data")
graph.add_edge("get_user_data", "loop_condition")
graph.add_conditional_edges(
    "loop_condition",
    lambda state: state["b"] != 0,
    {
        True: "modify",
        False: "write",
    },
)
graph.add_edge("modify", "loop_condition")
graph.add_edge("write", END)

workflow = graph.compile()

with open("graph1.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

workflow.invoke({}, {"recursion_limit": 100})

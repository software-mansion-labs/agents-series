from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.agents import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
from dotenv import load_dotenv


class State(AgentState):
    iteration: int


ITERATION_LIMIT = 5
load_dotenv()
model = init_chat_model("openai:gpt-4o")
model_with_search = model.bind_tools([DuckDuckGoSearchResults()])


def ask_llm(state: State) -> State:
    user_query = input("query: ")
    user_message = HumanMessage(user_query)
    answer_message: AIMessage = model_with_search.invoke(
        state["messages"] + [user_message]
    )

    return {
        "messages": [user_message, answer_message],
    }


def show_answer(state: State) -> State:
    print("answer: ", state["messages"][-1].content)

    return {
        "iteration": state["iteration"] + 1,
    }


def sum_up_search(state: State) -> State:
    answer_message: AIMessage = model.invoke(state["messages"])

    return {
        "messages": [answer_message],
    }


graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)
graph.add_node("web_search", ToolNode(tools=[DuckDuckGoSearchResults()]))
graph.add_node("show_answer", show_answer)
graph.add_node("sum_up_search", sum_up_search)

graph.add_edge(START, "ask_llm")
graph.add_conditional_edges(
    "ask_llm",
    tools_condition,
    {
        "tools": "web_search",
        END: "show_answer",
    },
)
graph.add_edge("web_search", "sum_up_search")
graph.add_edge("sum_up_search", "show_answer")
graph.add_conditional_edges(
    "show_answer",
    lambda state: state["iteration"] < ITERATION_LIMIT,
    {
        True: "ask_llm",
        False: END,
    },
)

workflow = graph.compile()

with open("graph5.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

workflow.invoke({"iteration": 0}, {"recursion_limit": 100})

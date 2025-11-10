from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import AgentState
from typing import Annotated, Sequence, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

loader = WebBaseLoader("https://docs.swmansion.com/react-native-executorch/")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

_ = vector_store.add_documents(documents=all_splits)


class State(AgentState):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int


class Decision(BaseModel):
    decision: Literal["yes", "no"] = Field(
        "whether the user specified to end the conversation (yes) or not (no)"
    )


ITERATION_LIMIT = 5
load_dotenv()
model = init_chat_model("openai:gpt-4o")
model_with_search = model.bind_tools([DuckDuckGoSearchResults()])
model_decision = model.with_structured_output(Decision)


def ask_llm(state: State) -> State:
    user_query = input("query: ")

    retrieved_docs = vector_store.similarity_search(user_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    user_message = HumanMessage(f"Context:\n{context}\n\nUser question:\n{user_query}")

    # homework: try to add an intermediate step, where a prompt is created by the LLM based on the query and context, before passing it to the LLM below.

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


def end_condition(state: State) -> Literal["yes", "no"]:
    decision: Decision = model_decision.invoke(
        state["messages"]
        + [SystemMessage("Did the user specify to end the conversation?")]
    )
    return decision.decision


def should_end(_: State) -> State:
    return {}


graph = StateGraph(State)

graph.add_node("ask_llm", ask_llm)
graph.add_node("web_search", ToolNode(tools=[DuckDuckGoSearchResults()]))
graph.add_node("show_answer", show_answer)
graph.add_node("sum_up_search", sum_up_search)
graph.add_node("should_end", should_end)

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
        True: "should_end",
        False: END,
    },
)
graph.add_conditional_edges(
    "should_end",
    end_condition,
    {
        "yes": END,
        "no": "ask_llm",
    },
)

checkpointer = InMemorySaver()  # used to store individual states
store = InMemoryStore()  # used to store states acrross threads
workflow = graph.compile(checkpointer=checkpointer, store=store)

with open("graph8.png", "wb") as f:
    f.write(workflow.get_graph().draw_mermaid_png())

config = {"recursion_limit": 100, "configurable": {"thread_id": "default-session"}}

# we start a thread
workflow.invoke(
    {"iteration": 0},
    config=config,
)

# now we resume the conversation (thread) in another invocation
workflow.invoke(
    workflow.get_state(config),  # last state of the agent
    config=config,
)

# homework: modify the agent so that it stores the state externally and loads the summarization or truncated history of your previous discussions when you run the script and resumes from there.

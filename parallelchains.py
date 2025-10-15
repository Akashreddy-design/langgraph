from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# ✅ instantiate the model properly
model = ChatOpenAI(model="gpt-4o-mini")

# ✅ define the state type
class LLM_state(TypedDict):
    question: str
    answer: str

# ✅ node function
def llm_qa(state: LLM_state) -> LLM_state:
    question = state["question"]
    prompt = f"Answer the following question: {question}"
    answer = model.invoke(prompt).content
    state["answer"] = answer
    return state

# ✅ build the workflow graph
graph = StateGraph(LLM_state)
graph.add_node("llm_qa", llm_qa)
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)
workflow = graph.compile()

# ✅ invoke
initial_state = {"question": "What is the distance between Earth and Moon?"}
final_state = workflow.invoke(initial_state)

print(final_state["answer"])

# 🦜🕸️ LangGraph — Zero to Advanced in 16 Days

A **hands-on, beginner-friendly** course: one small Jupyter notebook per day, from your very first graph to a multi-agent research assistant. Every lesson is short: **explain → run the code → check → try it yourself**. Every day ends with practice exercises and solutions.

> Built and tested with **LangGraph 1.2** and **LangChain 1.4** (Python 3.10+). The LLM is **OpenAI `gpt-4.1-nano`** — small, fast and very cheap.

---

## 🗺️ Course map

| Day | Notebook | You will learn | API key? |
|---|---|---|---|
| 🟢 1 | [Day01_Graph_Basics](Day01_Graph_Basics/Day01_Graph_Basics.ipynb) | State, nodes, edges, `START`/`END`, `compile`, `invoke`, `stream` | ❌ |
| 🟢 2 | [Day02_State_and_Reducers](Day02_State_and_Reducers/Day02_State_and_Reducers.ipynb) | Overwrite vs reducers, custom reducers, parallel nodes, input/output schemas | ❌ |
| 🟢 3 | [Day03_Branches_and_Loops](Day03_Branches_and_Loops/Day03_Branches_and_Loops.ipynb) | Conditional edges, `Literal` routers, loops, recursion limit, `Command` | ❌ |
| 🟡 4 | [Day04_LLM_and_Messages](Day04_LLM_and_Messages/Day04_LLM_and_Messages.ipynb) | `.env` setup, `ChatOpenAI`, messages, streaming tokens, LLM inside a node | ✅ |
| 🟡 5 | [Day05_Chatbot_MessagesState](Day05_Chatbot_MessagesState/Day05_Chatbot_MessagesState.ipynb) | `add_messages`, `MessagesState`, system prompts, multi-turn, `trim_messages` | ✅ |
| 🟡 6 | [Day06_Tools_and_Agents](Day06_Tools_and_Agents/Day06_Tools_and_Agents.ipynb) | `@tool`, `bind_tools`, ReAct loop by hand, `ToolNode`, `tools_condition`, `create_agent` | ✅ |
| 🟠 7 | [Day07_Memory_Checkpointers](Day07_Memory_Checkpointers/Day07_Memory_Checkpointers.ipynb) | Checkpointers, `thread_id`, `get_state`, history, SQLite memory, summarizing | ✅ |
| 🟠 8 | [Day08_Human_in_the_Loop](Day08_Human_in_the_Loop/Day08_Human_in_the_Loop.ipynb) | `interrupt()`, `Command(resume=)`, approvals, edits, `update_state`, time travel | ✅ |
| 🟠 9 | [Day09_Streaming_and_Async](Day09_Streaming_and_Async/Day09_Streaming_and_Async.ipynb) | Stream modes, token streaming, custom events, async & parallel speed-up | ✅ |
| 🔴 10 | [Day10_Structured_Output_and_Workflows](Day10_Structured_Output_and_Workflows/Day10_Structured_Output_and_Workflows.ipynb) | Structured output, chaining, routing, parallelization, evaluator–optimizer | ✅ |
| 🔴 11 | [Day11_Map_Reduce_with_Send](Day11_Map_Reduce_with_Send/Day11_Map_Reduce_with_Send.ipynb) | `Send`, map-reduce, orchestrator–workers | ✅ |
| 🔴 12 | [Day12_Subgraphs_and_Multi_Agent](Day12_Subgraphs_and_Multi_Agent/Day12_Subgraphs_and_Multi_Agent.ipynb) | Subgraphs, supervisor pattern, handoffs | ✅ |
| 🔴 13 | [Day13_Agentic_RAG](Day13_Agentic_RAG/Day13_Agentic_RAG.ipynb) | Embeddings, vector store, RAG graph, agentic & self-correcting RAG | ✅ |
| 🔴 14 | [Day14_Long_Term_Memory_Store](Day14_Long_Term_Memory_Store/Day14_Long_Term_Memory_Store.ipynb) | `Store`, runtime context, cross-thread memory, semantic memory search | ✅ |
| 🟣 15 | [Day15_Production_Skills](Day15_Production_Skills/Day15_Production_Skills.ipynb) | Retries, caching, fallbacks, Functional API, testing, LangSmith, deployment | ✅ |
| 🏆 16 | [Day16_Capstone_Research_Assistant](Day16_Capstone_Research_Assistant/Day16_Capstone_Research_Assistant.ipynb) | Everything together: plan → approve → parallel research → write → review | ✅ |

⏱️ Each day takes about **45–75 minutes**.

---

## ⚙️ Setup (5 minutes)

### 1. Get the code
```bash
git clone https://github.com/ajeetkumarAI/LangGraph-Handson-Notebook.git
cd LangGraph-Handson-Notebook
```

### 2. Create a virtual environment and install
```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add your OpenAI key (needed from Day 4)
Copy `.env.example` to `.env` **in the repo root** and paste your key:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-nano
```
Every notebook loads it with `load_dotenv(find_dotenv(usecwd=True))`, which searches the notebook's folder and its parents, so one `.env` in the root works for all days.

🔒 `.env` is in `.gitignore`, so it is never pushed to GitHub. **Never paste keys into notebooks.**

### 4. Open the notebooks
```bash
jupyter notebook          # or open the folder in VS Code
```
Start with **Day 1** and run the cells top to bottom with `Shift + Enter`.

---

## 📘 How each notebook is organised

```
Title + goal + lesson table
 └─ Setup cell (install + load .env)
 └─ Lesson 1: idea in plain English → small diagram → code → ✅ check → 🧪 try it
 └─ Lesson 2 …
 └─ Mini project (combines the day's lessons)
 └─ Practice exercises + ✅ solutions
 └─ 🎯 Recap + what's next
```

* `✅` cells contain small `assert` checks. If they run without errors, your code works.
* `show_graph(app)` draws each graph as a picture (needs internet). Offline, it prints Mermaid text you can paste into <https://mermaid.live>.

---

## 💰 Cost

All LLM lessons use `gpt-4.1-nano` with short prompts. The **whole course costs only a few rupees** of API usage. To use a smarter model (for example in the multi-agent lessons), set `OPENAI_MODEL=gpt-4.1-mini` in `.env`.

---

## 🧠 The 5-step recipe you'll use every day

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):                 # 1. State
    text: str

def shout(state: State) -> dict:        # 2. Node
    return {"text": state["text"].upper()}

builder = StateGraph(State)             # 3. Builder + nodes
builder.add_node("shout", shout)
builder.add_edge(START, "shout")        # 4. Edges
builder.add_edge("shout", END)

app = builder.compile()                 # 5. Compile + run
print(app.invoke({"text": "hello langgraph"}))
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY not found` | `.env` must be in the repo root (next to this README) and the line must be `OPENAI_API_KEY=sk-...` with no quotes or spaces |
| `ModuleNotFoundError` | activate your venv and run `pip install -r requirements.txt` again, then restart the kernel |
| Graph picture doesn't show | you are offline, so use the printed Mermaid text instead |
| `GraphRecursionError` | your loop has no exit. Check your router, or raise `recursion_limit` |
| An agent behaves oddly | nano is a tiny model, so try `OPENAI_MODEL=gpt-4.1-mini` |

---

## 👨‍🏫 Trainer

Created by **Ajeetkumar** — GenAI Engineer. ⭐ Star the repo if it helps you learn!

📜 License: see [LICENSE](LICENSE).

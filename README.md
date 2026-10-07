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

## 🚀 Quick start (for learners)

**Prerequisites:** Python **3.10+**, Git, and an OpenAI API key (only needed from Day 4: <https://platform.openai.com/api-keys>).

### Step 1 — Clone the repository
```bash
git clone https://github.com/ajeetkumarAI/LangGraph-Handson-Notebook.git
cd LangGraph-Handson-Notebook
```
No Git? On the GitHub page, click **Code → Download ZIP** and unzip it.

### Step 2 — Create an environment and install the packages

**Option A: venv (works everywhere)**

Windows (Command Prompt / PowerShell):
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Option B: Anaconda**
```bash
conda create -n langgraph python=3.11 -y
conda activate langgraph
pip install -r requirements.txt
```

✅ Check it worked:
```bash
python -c "import langgraph, langchain_openai; print('Ready!')"
```

### Step 3 — Add your OpenAI key (from Day 4)

Make a copy of `.env.example` and name it **`.env`** (in the main repo folder, next to this README):

```bash
# Windows
copy .env.example .env
# Mac / Linux
cp .env.example .env
```

Open `.env` in any editor and paste your key:
```
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4.1-nano
```

Every notebook loads it automatically with `load_dotenv(find_dotenv(usecwd=True))`. It searches the notebook's folder **and its parent folders**, so one `.env` in the root works for all 16 days.

🔒 `.env` is listed in `.gitignore`, so it is never pushed to GitHub. **Never paste your key into a notebook.**

### Step 4 — Open and run the notebooks

**Jupyter:**
```bash
jupyter notebook
```
Your browser opens. Go to `Day01_Graph_Basics/` → open `Day01_Graph_Basics.ipynb`.

**VS Code:** open the repo folder → open a notebook → click **Select Kernel** (top right) → choose the `.venv` (or `langgraph` conda) environment.

Run cells **top to bottom** with `Shift + Enter`. Each notebook has a `%pip install` cell at the top, so it also works on **Google Colab** (upload the notebook, then add your key with `os.environ["OPENAI_API_KEY"] = ...` in a private cell, or use Colab Secrets).

### Step 5 — Follow the days in order

```
Day01 → Day02 → Day03      core graph skills, no API key, free
Day04 → … → Day09          LLMs, chatbots, tools, memory, human-in-the-loop, streaming
Day10 → … → Day16          advanced patterns, multi-agent, RAG, production, capstone
```

For each day: read → run → do the **🧪 Try it** prompts → solve the **Practice** exercises before you look at the solutions.

### Getting updates
```bash
git pull
```
(If you edited a notebook, save your copy under a new name first, e.g. `Day05_my_notes.ipynb`, to avoid merge conflicts.)

---

## 📁 Repository structure

```
LangGraph-Handson-Notebook/
├── README.md                  ← you are here
├── requirements.txt           ← all packages for all 16 days
├── .env.example               ← copy to .env and add your key
├── Day01_Graph_Basics/
│   └── Day01_Graph_Basics.ipynb
├── Day02_State_and_Reducers/
│   └── ...
├── ...
├── Day16_Capstone_Research_Assistant/
└── archive/                   ← older versions of the notebooks
```

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

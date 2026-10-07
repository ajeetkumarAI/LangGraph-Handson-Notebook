# LangGraph from Zero to Advanced: a 16-Day Hands-On Course

This repository is a step-by-step course on LangGraph, written the way I teach it in class. There's one Jupyter notebook per day. We start with plain Python graphs (no API key, no cost) and finish with a multi-agent research assistant.

Each notebook is meant to be *worked through*, not just read:

- **Plain explanations first.** Every idea starts with an everyday comparison and a short note on why it matters, before any code.
- **Predict, then run.** Before key cells you're asked to guess the output. The answer is hidden in a click-to-expand box, so you can check yourself.
- **"Your turn" cells.** Small pieces of starter code where you change one thing and see what happens.
- **Common mistakes.** The errors students actually run into, shown live, with the fix.
- **Small cells.** One function, one class or one building step per cell, so you can run and understand one piece at a time.
- **Practice exercises** at the end of every day, with solutions.

The notebooks were written and tested with LangGraph 1.2 and LangChain 1.4 on Python 3.10 and above. The LLM lessons use OpenAI's `gpt-4.1-nano`, which is small, fast and very cheap.

---

## Course outline

| Day | Notebook | What you'll learn | API key needed? |
|---|---|---|---|
| 1 | [Your First Graph](Day01_Graph_Basics/Day01_Graph_Basics.ipynb) | State, nodes, edges, START and END, compile, invoke, stream | No |
| 2 | [State and Reducers](Day02_State_and_Reducers/Day02_State_and_Reducers.ipynb) | Overwrite vs. combine, custom reducers, parallel nodes, input/output schemas | No |
| 3 | [Branches and Loops](Day03_Branches_and_Loops/Day03_Branches_and_Loops.ipynb) | Routers, `Literal`, loops, recursion limit, `Command` | No |
| 4 | [Talking to an LLM](Day04_LLM_and_Messages/Day04_LLM_and_Messages.ipynb) | `.env` setup, `ChatOpenAI`, message types, streaming, an LLM inside a node | Yes |
| 5 | [Building a Chatbot](Day05_Chatbot_MessagesState/Day05_Chatbot_MessagesState.ipynb) | `add_messages`, `MessagesState`, system prompts, multi-turn chat, trimming | Yes |
| 6 | [Tools and Your First Agent](Day06_Tools_and_Agents/Day06_Tools_and_Agents.ipynb) | `@tool`, `bind_tools`, the agent loop by hand, `ToolNode`, `create_agent` | Yes |
| 7 | [Memory](Day07_Memory_Checkpointers/Day07_Memory_Checkpointers.ipynb) | Checkpointers, thread ids, `get_state`, SQLite memory, summarizing long chats | Yes |
| 8 | [Human in the Loop](Day08_Human_in_the_Loop/Day08_Human_in_the_Loop.ipynb) | `interrupt()`, approvals, edits, approving tool calls, time travel | Yes |
| 9 | [Streaming and Async](Day09_Streaming_and_Async/Day09_Streaming_and_Async.ipynb) | Stream modes, token streaming, custom progress, async and parallel calls | Yes |
| 10 | [Structured Output and Workflow Patterns](Day10_Structured_Output_and_Workflows/Day10_Structured_Output_and_Workflows.ipynb) | Pydantic output, chaining, routing, parallelization, evaluator loops | Yes |
| 11 | [Map-Reduce with Send](Day11_Map_Reduce_with_Send/Day11_Map_Reduce_with_Send.ipynb) | `Send`, map-reduce, orchestrator and workers | Yes |
| 12 | [Subgraphs and Multi-Agent Systems](Day12_Subgraphs_and_Multi_Agent/Day12_Subgraphs_and_Multi_Agent.ipynb) | Subgraphs, a supervisor team, agent handoffs | Yes |
| 13 | [Agentic RAG](Day13_Agentic_RAG/Day13_Agentic_RAG.ipynb) | Embeddings, vector stores, RAG, agentic and self-correcting RAG | Yes |
| 14 | [Long-Term Memory](Day14_Long_Term_Memory_Store/Day14_Long_Term_Memory_Store.ipynb) | The Store, runtime context, remembering users across chats | Yes |
| 15 | [Getting Ready for the Real World](Day15_Production_Skills/Day15_Production_Skills.ipynb) | Retries, caching, fallbacks, Functional API, testing, LangSmith, deployment | Yes |
| 16 | [Capstone: a Research Assistant](Day16_Capstone_Research_Assistant/Day16_Capstone_Research_Assistant.ipynb) | One complete project using almost everything above | Yes |

Plan for about 45 to 75 minutes per day.

---

## Getting started

You'll need Python 3.10 or newer and Git. From Day 4 onwards you'll also need an OpenAI API key, which you can create at https://platform.openai.com/api-keys.

### Step 1: Clone the repository

```bash
git clone https://github.com/ajeetkumarAI/LangGraph-Handson-Notebook.git
cd LangGraph-Handson-Notebook
```

If you don't use Git, open the repository on GitHub, click **Code**, then **Download ZIP**, and unzip it.

### Step 2: Create an environment and install the packages

**Option A: venv**

On Windows (Command Prompt or PowerShell):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Mac or Linux:

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

To check that everything installed correctly:

```bash
python -c "import langgraph, langchain_openai; print('Ready')"
```

### Step 3: Add your OpenAI key (needed from Day 4)

Make a copy of `.env.example` called `.env`, in the main folder of the repository (next to this README):

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` in any text editor and paste your key:

```
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4.1-nano
```

Every notebook loads this file automatically. It searches the notebook's own folder and then the parent folders, so one `.env` in the main folder works for all 16 days.

Please don't paste your key into a notebook. `.env` is listed in `.gitignore`, so it never gets uploaded to GitHub. A notebook, on the other hand, is easy to share by accident.

### Step 4: Open the notebooks

**Jupyter:**

```bash
jupyter notebook
```

Your browser will open. Go into `Day01_Graph_Basics` and open `Day01_Graph_Basics.ipynb`.

**VS Code:** open the repository folder, open a notebook, click **Select Kernel** in the top right, and choose your `.venv` (or the `langgraph` conda environment).

**Google Colab:** upload a notebook. Each one has a `%pip install` cell at the top. Add your key with Colab's **Secrets** panel rather than typing it into a cell.

Run the cells from top to bottom with `Shift + Enter`.

### Step 5: Work through the days in order

```
Days 1 to 3     The core of LangGraph, with plain Python. No API key, no cost.
Days 4 to 9     LLMs, chatbots, tools, memory, human approval, streaming.
Days 10 to 16   Advanced patterns, multi-agent systems, RAG, production, capstone.
```

A suggestion from experience: when a notebook asks you to predict an output, actually stop and guess before opening the answer. And try the practice exercises before looking at the solutions. That's where most of the learning happens.

### Getting updates

```bash
git pull
```

If you've edited a notebook, save your copy under a different name first (for example `Day05_my_notes.ipynb`), so your changes don't clash with the update.

---

## Repository layout

```
LangGraph-Handson-Notebook/
    README.md               this file
    requirements.txt        all the packages for all 16 days
    .env.example            copy this to .env and add your key
    Day01_Graph_Basics/
        Day01_Graph_Basics.ipynb
    Day02_State_and_Reducers/
    ...
    Day16_Capstone_Research_Assistant/
    archive/                older versions of the notebooks
```

---

## The pattern you'll use every day

Every graph in this course is built with the same five steps:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Describe the state
class State(TypedDict):
    text: str

# 2. Write a node (a plain function)
def shout(state: State) -> dict:
    return {"text": state["text"].upper()}

# 3. Create a builder and add the node
builder = StateGraph(State)
builder.add_node("shout", shout)

# 4. Connect it with edges
builder.add_edge(START, "shout")
builder.add_edge("shout", END)

# 5. Compile and run
app = builder.compile()
print(app.invoke({"text": "hello langgraph"}))
```

---

## Cost

All the LLM lessons use `gpt-4.1-nano` with short prompts, so working through the whole course costs only a few rupees in API usage. If a lesson needs a smarter model (the multi-agent day is the most demanding), set `OPENAI_MODEL=gpt-4.1-mini` in `.env`.

---

## Troubleshooting

| Problem | What to check |
|---|---|
| `OPENAI_API_KEY not found` | `.env` must be in the main folder, next to this README. The line must be exactly `OPENAI_API_KEY=sk-...`, with no quotes or spaces. On Windows, make sure the file isn't secretly called `.env.txt`. Restart the kernel after creating it. |
| `ModuleNotFoundError` | Activate your environment, run `pip install -r requirements.txt` again, and restart the kernel. |
| The graph picture doesn't appear | Drawing the picture needs internet. Offline, the notebook prints the diagram as text, which you can paste into https://mermaid.live. |
| `GraphRecursionError` | A loop never reached its exit. Check your router's condition. |
| An agent behaves strangely | `gpt-4.1-nano` is a very small model. Try `OPENAI_MODEL=gpt-4.1-mini`. |

---

## About

Course written by **Ajeetkumar**, GenAI Engineer. If it helped you, a star on the repository is always appreciated.

License: see [LICENSE](LICENSE).

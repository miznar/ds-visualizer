import os
import re
import io
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="LLM Data Structures Visualizer (MVP)", page_icon="🧠", layout="wide")

st.markdown(
    """
    <h1 style="text-align:center">🧠 LLM-Powered Data Structures Visualizer (MVP)</h1>
    <p style="text-align:center; color:gray; font-size:16px;">
        Paste simple code-like commands (push/pop, enqueue/dequeue, insert), step through the simulation,
        watch the structure update, and read explanations.<br>
        This MVP supports <b>Stack</b>, <b>Queue</b>, and <b>BST (insert)</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ==============================
# Simulators
# ==============================

class StackSim:
    def __init__(self):
        self.stack: List[Any] = []

    def apply(self, op: str, arg: Optional[Any] = None) -> str:
        before = self.state()
        if op == "push" and arg is not None:
            self.stack.append(arg)
        elif op == "pop":
            if self.stack:
                self.stack.pop()
        after = self.state()
        return explanation_stack(op, arg, before, after)

    def state(self):
        return list(self.stack)


class QueueSim:
    def __init__(self):
        self.q: List[Any] = []

    def apply(self, op: str, arg: Optional[Any] = None) -> str:
        before = self.state()
        if op in ("enqueue", "push") and arg is not None:
            self.q.append(arg)
        elif op in ("dequeue", "pop"):
            if self.q:
                self.q.pop(0)
        after = self.state()
        return explanation_queue(op, arg, before, after)

    def state(self):
        return list(self.q)


# --- BST (insert only MVP) ---
@dataclass
class Node:
    key: int
    left: Optional['Node'] = None
    right: Optional['Node'] = None

class BSTSim:
    def __init__(self):
        self.root: Optional[Node] = None

    def insert(self, key: int):
        if self.root is None:
            self.root = Node(key)
            return [f"Tree empty → make {key} the root."]
        steps = []
        cur = self.root
        while True:
            if key < cur.key:
                steps.append(f"{key} < {cur.key} → go left")
                if cur.left is None:
                    cur.left = Node(key)
                    steps.append(f"Insert {key} as left child of {cur.key}")
                    break
                cur = cur.left
            elif key > cur.key:
                steps.append(f"{key} > {cur.key} → go right")
                if cur.right is None:
                    cur.right = Node(key)
                    steps.append(f"Insert {key} as right child of {cur.key}")
                    break
                cur = cur.right
            else:
                steps.append(f"{key} == {cur.key} → duplicate, ignore (MVP)")
                break
        return steps

    def apply(self, op: str, arg: Optional[Any] = None) -> str:
        before = self.as_edges()
        narrative = []
        if op == "insert" and isinstance(arg, int):
            narrative = self.insert(arg)
        after = self.as_edges()
        return explanation_bst(op, arg, before, after, narrative)

    def as_edges(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int,int]] = []
        def dfs(n: Optional[Node]):
            if not n: return
            if n.left:
                edges.append((n.key, n.left.key))
                dfs(n.left)
            if n.right:
                edges.append((n.key, n.right.key))
                dfs(n.right)
        dfs(self.root)
        return edges

# ==============================
# Visualizations
# ==============================

def draw_stack(stack: List[Any]):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(range(len(stack)), stack)
    ax.set_title("Stack (top is right)")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    st.pyplot(fig, use_container_width=True)


def draw_queue(q: List[Any]):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(range(len(q)), q)
    ax.set_title("Queue (front is left)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    st.pyplot(fig, use_container_width=True)


def draw_bst(edges: List[Tuple[int, int]]):
    # Build graph
    G = nx.DiGraph()
    for u, v in edges:
        G.add_edge(u, v)
    # Try to make a hierarchy layout (simple)
    try:
        pos = hierarchy_pos(G)
    except Exception:
        pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, arrows=False, node_size=1200)
    ax.set_title("BST")
    st.pyplot(fig, use_container_width=True)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    # Adapted from networkx docs (simple tree layout)
    if not nx.is_tree(G):
        return nx.spring_layout(G)
    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter([n for n, d in G.in_degree() if d == 0]), None)
        else:
            root = list(G.nodes)[0]
    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, xcenter, pos, parent=None):
        children = list(G.successors(root)) if isinstance(G, nx.DiGraph) else list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children = [c for c in children if c != parent]
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, leftmost, dx, vert_gap, vert_loc - vert_gap, nextx, pos, root)
        pos[root] = (xcenter, vert_loc)
        return pos
    return _hierarchy_pos(G, root, 0, width, vert_gap, vert_loc, xcenter, {})

# ==============================
# LLM / Explanations (MVP: fallback + optional OpenAI)
# ==============================

SYSTEM_PROMPT = (
    "You are a concise teaching assistant. Explain each step of a data structure operation in simple terms."
)
# Utility: unified state getter
def get_state(sim, ds_choice):
    if ds_choice == "Stack":
        return sim.state()
    elif ds_choice == "Queue":
        return sim.state()
    elif ds_choice == "BST":
        return sim.as_edges()  # or sim.state() if you implemented it that way
    else:
        return None


def have_openai() -> bool:
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())
    return bool(key)


def llm_explain(messages: List[Dict[str,str]]) -> Optional[str]:
    """Optional OpenAI call. If no key, return None to use fallback."""
    if not have_openai():
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


# --- Rule-based fallbacks ---

def explanation_stack(op, arg, before, after) -> str:
    base = f"Operation: {op}{'(' + str(arg) + ')' if arg is not None else ''}.\n"
    if op == "push":
        tip = "Pushed to the top."
    elif op == "pop":
        tip = "Removed from the top (if not empty)."
    else:
        tip = "Unknown op (MVP supports push/pop)."
    return base + f"Before: {before}\nAfter: {after}\nExplanation: {tip}"


def explanation_queue(op, arg, before, after) -> str:
    base = f"Operation: {op}{'(' + str(arg) + ')' if arg is not None else ''}.\n"
    if op in ("enqueue", "push"):
        tip = "Added to the back of the queue."
    elif op in ("dequeue", "pop"):
        tip = "Removed from the front (if not empty)."
    else:
        tip = "Unknown op (MVP supports enqueue/dequeue)."
    return base + f"Before: {before}\nAfter: {after}\nExplanation: {tip}"


def explanation_bst(op, arg, before_edges, after_edges, path_steps: List[str]) -> str:
    base = f"Operation: {op}({arg}).\n"
    path = " → ".join(path_steps) if path_steps else "(no change)"
    return base + f"Edges before: {before_edges}\nEdges after: {after_edges}\nPath: {path}"


# ==============================
# Parsing simple commands from code-like input
# ==============================

COMMAND_PATTERNS = [
    ("push", re.compile(r"push\(([-+]?\d+)\)")),
    ("pop", re.compile(r"pop\(\)")),
    ("enqueue", re.compile(r"enqueue\(([-+]?\d+)\)")),
    ("dequeue", re.compile(r"dequeue\(\)")),
    ("insert", re.compile(r"insert\(([-+]?\d+)\)")),
]


def extract_commands(code: str) -> List[Tuple[str, Optional[int]]]:
    lines = [l.strip() for l in code.splitlines() if l.strip()]
    ops: List[Tuple[str, Optional[int]]] = []
    for ln in lines:
        matched = False
        for name, pat in COMMAND_PATTERNS:
            m = pat.search(ln)
            if m:
                arg = int(m.group(1)) if m.groups() else None
                ops.append((name, arg))
                matched = True
                break
        if not matched:
            # try simple assignments like x=5 then push(x)
            m_assign = re.search(r"([a-zA-Z_]\w*)\s*=\s*([-+]?\d+)", ln)
            m_push_var = re.search(r"push\(([_a-zA-Z]\w*)\)", ln)
            if m_assign and m_push_var and m_assign.group(1) == m_push_var.group(1):
                ops.append(("push", int(m_assign.group(2))))
    return ops


# ==============================
# Default examples
# ==============================

EXAMPLES = {
    "Stack": """# Stack example\npush(3)\npush(7)\npop()\npush(10)\n""",
    "Queue": """# Queue example\nenqueue(5)\nenqueue(6)\ndequeue()\nenqueue(9)\n""",
    "BST": """# BST insert example\ninsert(8)\ninsert(3)\ninsert(10)\ninsert(1)\ninsert(6)\ninsert(14)\ninsert(4)\ninsert(7)\n""",
}

# ==============================
# UI: Sidebar controls
# ==============================

st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", use_container_width=True)
st.sidebar.title("⚙️ Settings")

DS_TYPES = ["Stack", "Queue", "BST"]
ds_choice = st.sidebar.selectbox("Choose Data Structure", DS_TYPES, index=0)

use_openai = st.sidebar.toggle("Use OpenAI for explanations (if key present)", value=False, help="Reads OPENAI_API_KEY from environment.")

st.sidebar.caption(f"OpenAI key detected: {'✅' if have_openai() else '❌'}")

# ==============================
# UI: Code input + controls
# ==============================

st.markdown("### ✍️ Input Code / Commands")
code = st.text_area("Paste your commands (e.g., push(5), pop(), insert(10)):", value=EXAMPLES[ds_choice], height=220)

ops = extract_commands(code)

if "session" not in st.session_state:
    st.session_state.session = {
        "ops": ops,
        "step": 0,
        "history": [],  # list of dicts: {op,arg,state_before,state_after,explanation}
    }

# Reset if code changed
prev_ops = st.session_state.session.get("ops", [])
if prev_ops != ops:
    st.session_state.session = {"ops": ops, "step": 0, "history": []}

# Instantiate simulator
if ds_choice == "Stack":
    sim = StackSim()
elif ds_choice == "Queue":
    sim = QueueSim()
else:
    sim = BSTSim()

# Rebuild up to current step from history
for h in st.session_state.session["history"]:
    # reapply to reach prior state
    if ds_choice == "Stack":
        if h["op"] == "push": sim.apply("push", h["arg"])  # already applied, but idempotent enough for MVP
        elif h["op"] == "pop": sim.apply("pop")
    elif ds_choice == "Queue":
        if h["op"] in ("enqueue", "push"): sim.apply("enqueue", h["arg"])  # enqueue == push (MVP)
        elif h["op"] in ("dequeue", "pop"): sim.apply("dequeue")
    else:
        if h["op"] == "insert": sim.apply("insert", h["arg"])  # BST

# Controls
colA, colB, colC, colD = st.columns([1,1,1,2])
with colA:
    if st.button("⏮️ Reset"):
        st.session_state.session = {"ops": ops, "step": 0, "history": []}
        st.rerun()
with colB:
    if st.button("▶️ Next Step"):
        i = st.session_state.session["step"]
        if i < len(ops):
            op, arg = ops[i]
            # Take snapshot before
            before_state = get_state(sim, ds_choice)
            # Apply
            explanation_text = sim.apply(op, arg)
            # Optional LLM augmentation
            if use_openai:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Data Structure: {ds_choice}\nOperation: {op}({arg})\nBefore: {before_state}\nAfter: {get_state(sim, ds_choice)}"}
                ]
                model_out = llm_explain(messages)
                if model_out:
                    explanation_text += "\n\nLLM: " + model_out
            # Save history
            st.session_state.session["history"].append({
                "op": op, "arg": arg,
                "before": before_state,
                "after": get_state(sim, ds_choice),
                "explanation": explanation_text,
            })
            st.session_state.session["step"] += 1
            st.rerun()
with colC:
    if st.button("⏸️ Step Back"):
        # Rebuild fresh up to step-1
        i = st.session_state.session["step"]
        if i > 0:
            st.session_state.session["step"] -= 1
            # Trim history
            st.session_state.session["history"] = st.session_state.session["history"][:-1]
            # Hard reset and replay
            st.rerun()
with colD:
    if st.button("⬇️ Download Session JSON", use_container_width=True):
        buf = io.StringIO()
        json.dump(st.session_state.session, buf, indent=2)
        st.download_button("Download", buf.getvalue(), file_name="session.json")

# Layout: Visualization and Explanation
left, right = st.columns([1,1])
with left:
    st.subheader("📊 Visualization")
    if ds_choice == "Stack":
        draw_stack(get_state(sim, ds_choice))
    elif ds_choice == "Queue":
        draw_queue(get_state(sim, ds_choice))
    else:
        draw_bst(sim.as_edges())

with right:
    st.subheader("🧠 Explanations (per step)")
    if st.session_state.session["history"]:
        for idx, h in enumerate(st.session_state.session["history"], start=1):
            st.markdown(f"**Step {idx}:** `{h['op']}` {'' if h['arg'] is None else h['arg']}")
            st.code(h["explanation"], language="text")
    else:
        st.info("Run steps to see explanations here.")

# ==============================
# Helpers
# ==============================

def get_state(sim_obj, ds_type: str):
    if ds_type == "Stack":
        return sim_obj.state()
    if ds_type == "Queue":
        return sim_obj.state()
    # BST → return edges list for a stable summary
    return sim_obj.as_edges()

# ==============================
# Footer / Tips
# ==============================

st.markdown(
    """
    <hr>
    <p style="color:gray; font-size:14px;">
    Tips:
    <ul>
        <li>Use commands like <code>push(5)</code>, <code>pop()</code>, <code>enqueue(7)</code>, <code>dequeue()</code>, <code>insert(10)</code>.</li>
        <li>Toggle OpenAI in the sidebar to augment rule-based explanations (requires <code>OPENAI_API_KEY</code>).</li>
        <li>This is an MVP: extend by adding delete for BST, AVL rotations, and parsing of real Python code via AST.</li>
    </ul>
    </p>
    """,
    unsafe_allow_html=True,
)

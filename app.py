import os
import re
import io
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
import time

import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="LLM Data Structures Visualizer (Enhanced)", page_icon="🧠", layout="wide")

st.markdown("""
<div style="text-align:center; padding:10px; border-radius:12px; background:linear-gradient(90deg,#f4f6fb,#9fb7da);">
<h1 style="color:#192438;">🧠 LLM-Powered Data Structures Visualizer</h1>
<p style="color:#263754; font-size:16px;">
Step through Stack, Queue, or BST operations and watch visualizations update.<br>
Optional LLM explanations for each step.
</p>
</div>
<hr>
""", unsafe_allow_html=True)

# ==============================
# Simulators
# ==============================
class StackSim:
    def __init__(self): self.stack: List[Any] = []
    def apply(self, op: str, arg: Optional[Any] = None) -> str:
        before = self.state()
        if op == "push" and arg is not None: self.stack.append(arg)
        elif op == "pop" and self.stack: self.stack.pop()
        after = self.state()
        return f"Operation: {op}({arg})\nBefore: {before}\nAfter: {after}\nTip: {'Pushed to top' if op=='push' else 'Removed from top'}"
    def state(self): return list(self.stack)

class QueueSim:
    def __init__(self): self.q: List[Any] = []
    def apply(self, op: str, arg: Optional[Any] = None) -> str:
        before = self.state()
        if op in ("enqueue","push") and arg is not None: self.q.append(arg)
        elif op in ("dequeue","pop") and self.q: self.q.pop(0)
        after = self.state()
        return f"Operation: {op}({arg})\nBefore: {before}\nAfter: {after}\nTip: {'Added to back' if op in ('enqueue','push') else 'Removed from front'}"
    def state(self): return list(self.q)

@dataclass
class Node: key:int; left:Optional['Node']=None; right:Optional['Node']=None
class BSTSim:
    def __init__(self): self.root: Optional[Node]=None
    def insert(self, key:int):
        if self.root is None: self.root=Node(key); return [f"Tree empty → make {key} root"]
        steps=[]; cur=self.root
        while True:
            if key<cur.key: steps.append(f"{key} < {cur.key} → left"); 
            if cur.left is None: cur.left=Node(key); steps.append(f"Insert {key} left of {cur.key}"); break; cur=cur.left
            elif key>cur.key: steps.append(f"{key} > {cur.key} → right"); 
            if cur.right is None: cur.right=Node(key); steps.append(f"Insert {key} right of {cur.key}"); break; cur=cur.right
            else: steps.append(f"{key} == {cur.key} → ignore duplicate"); break
        return steps
    def apply(self, op:str, arg:Optional[Any]=None):
        before=self.as_edges(); narrative=[]
        if op=="insert" and isinstance(arg,int): narrative=self.insert(arg)
        after=self.as_edges()
        path = " → ".join(narrative) if narrative else "(no change)"
        return f"Operation: {op}({arg})\nEdges before: {before}\nEdges after: {after}\nPath: {path}"
    def as_edges(self)->List[Tuple[int,int]]:
        edges=[]; 
        def dfs(n:Optional[Node]):
            if not n: return
            if n.left: edges.append((n.key,n.left.key)); dfs(n.left)
            if n.right: edges.append((n.key,n.right.key)); dfs(n.right)
        dfs(self.root); return edges

# ==============================
# Visualizations
# ==============================
def draw_stack(stack): fig, ax = plt.subplots(figsize=(4,4)); ax.bar(range(len(stack)),stack); ax.set_title("Stack (top → right)"); st.pyplot(fig,use_container_width=True)
def draw_queue(q): fig, ax = plt.subplots(figsize=(4,4)); ax.bar(range(len(q)),q); ax.set_title("Queue (front → left)"); st.pyplot(fig,use_container_width=True)
def draw_bst(edges):
    G=nx.DiGraph(); [G.add_edge(u,v) for u,v in edges]
    try: pos=hierarchy_pos(G)
    except: pos=nx.spring_layout(G)
    fig,ax=plt.subplots(figsize=(5,4)); nx.draw(G,pos,with_labels=True,node_size=1200, arrows=False); st.pyplot(fig,use_container_width=True)
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G): return nx.spring_layout(G)
    if root is None: root = next(iter([n for n,d in G.in_degree() if d==0]), None)
    def _hierarchy_pos(G, root, leftmost, width, vert_gap, vert_loc, xcenter, pos, parent=None):
        children=list(G.successors(root)) if isinstance(G,nx.DiGraph) else list(G.neighbors(root))
        if not isinstance(G,nx.DiGraph) and parent: children=[c for c in children if c!=parent]
        if children:
            dx=width/len(children); nextx=xcenter-width/2-dx/2
            for c in children: nextx+=dx; pos=_hierarchy_pos(G,c,leftmost,dx,vert_gap,vert_loc-vert_gap,nextx,pos,root)
        pos[root]=(xcenter,vert_loc); return pos
    return _hierarchy_pos(G,root,0,width,vert_gap,vert_loc,xcenter,{})

# ==============================
# Parsing
# ==============================
COMMAND_PATTERNS=[("push",re.compile(r"push\(([-+]?\d+)\)")),("pop",re.compile(r"pop\(\)")),("enqueue",re.compile(r"enqueue\(([-+]?\d+)\)")),("dequeue",re.compile(r"dequeue\(\)")),("insert",re.compile(r"insert\(([-+]?\d+)\)"))]
def extract_commands(code:str)->List[Tuple[str,Optional[int]]]:
    lines=[l.strip() for l in code.splitlines() if l.strip()]; ops=[]
    for ln in lines:
        matched=False
        for name,pat in COMMAND_PATTERNS:
            m=pat.search(ln)
            if m: ops.append((name,int(m.group(1)) if m.groups() else None)); matched=True; break
        if not matched:
            m_assign=re.search(r"([a-zA-Z_]\w*)\s*=\s*([-+]?\d+)",ln)
            m_push_var=re.search(r"push\(([_a-zA-Z]\w*)\)",ln)
            if m_assign and m_push_var and m_assign.group(1)==m_push_var.group(1): ops.append(("push",int(m_assign.group(2))))
    return ops

# ==============================
# Defaults / UI
# ==============================
EXAMPLES={"Stack":"push(3)\npush(7)\npop()\npush(10)","Queue":"enqueue(5)\nenqueue(6)\ndequeue()\nenqueue(9)","BST":"insert(8)\ninsert(3)\ninsert(10)\ninsert(1)\ninsert(6)\ninsert(14)\ninsert(4)\ninsert(7)"}

st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", use_container_width=True)
st.sidebar.title("⚙️ Settings")
DS_TYPES=["Stack","Queue","BST"]
ds_choice=st.sidebar.selectbox("Choose Data Structure",DS_TYPES,index=0)
use_openai=st.sidebar.toggle("Use OpenAI for explanations",value=False)
st.sidebar.caption(f"OpenAI key detected: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")

st.markdown("### ✍️ Input Commands")
code=st.text_area("Commands:",value=EXAMPLES[ds_choice],height=180)
ops=extract_commands(code)

if "session" not in st.session_state: st.session_state.session={"ops":ops,"step":0,"history":[]}
if st.session_state.session.get("ops",[])!=ops: st.session_state.session={"ops":ops,"step":0,"history":[]}

sim={"Stack":StackSim(),"Queue":QueueSim(),"BST":BSTSim()}[ds_choice]

# Replay history to reach current step
for h in st.session_state.session["history"]:
    if ds_choice=="Stack": sim.apply(h["op"],h["arg"])
    elif ds_choice=="Queue": sim.apply(h["op"],h["arg"])
    else: sim.apply(h["op"],h["arg"])

# Controls
colA,colB,colC,colD=st.columns([1,1,1,2])
with colA:
    if st.button("⏮️ Reset"): st.session_state.session={"ops":ops,"step":0,"history":[]}; st.rerun()
with colB:
    if st.button("▶️ Next Step"):
        i=st.session_state.session["step"]
        if i<len(ops):
            op,arg=ops[i]; before=getattr(sim,'state',lambda:sim.as_edges())()
            explanation=sim.apply(op,arg)
            if use_openai: explanation+="\nLLM: (simulated)"  # placeholder
            st.session_state.session["history"].append({"op":op,"arg":arg,"before":before,"after":getattr(sim,'state',lambda:sim.as_edges())(),"explanation":explanation})
            st.session_state.session["step"]+=1; st.rerun()
with colC:
    if st.button("⏸️ Step Back"):
        i=st.session_state.session["step"]
        if i>0: st.session_state.session["step"]-=1; st.session_state.session["history"]=st.session_state.session["history"][:-1]; st.rerun()
with colD:
    if st.button("⬇️ Download JSON",use_container_width=True):
        buf=io.StringIO(); json.dump(st.session_state.session,buf,indent=2); st.download_button("Download",buf.getvalue(),"session.json")

# Visualization + Explanations
left,right=st.columns([1,1])
with left: st.subheader("📊 Visualization"); 
if ds_choice=="Stack": draw_stack(getattr(sim,'state')())
elif ds_choice=="Queue": draw_queue(getattr(sim,'state')())
else: draw_bst(sim.as_edges())

with right:
    st.subheader("🧠 Explanations")
    if st.session_state.session["history"]:
        for idx,h in enumerate(st.session_state.session["history"],start=1):
            st.markdown(f"<div style='padding:8px; border-radius:10px; background:#e8ecf6; margin-bottom:6px;'><b>Step {idx}:</b> {h['op']} {'' if h['arg'] is None else h['arg']}</div>",unsafe_allow_html=True)
            st.code(h["explanation"],language="text")
    else: st.info("Run steps to see explanations here.")

# Footer / Tips
st.markdown("""
<hr>
<p style="color:gray; font-size:14px;">
Tips:
<ul>
<li>Commands: <code>push(5)</code>, <code>pop()</code>, <code>enqueue(7)</code>, <code>dequeue()</code>, <code>insert(10)</code></li>
<li>Toggle OpenAI to augment explanations</li>
<li>Extend MVP: BST delete, AVL rotations, real Python AST parsing</li>
</ul>
</p>
""",unsafe_allow_html=True)

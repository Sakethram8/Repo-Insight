# app.py
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import falkordb
import time
import openai
from collections import Counter
from tools import get_macro_architecture, get_class_architecture
from config import FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME, SGLANG_BASE_URL, SGLANG_API_KEY, LLM_MODEL
from agent import run_repo_agent

st.set_page_config(page_title="Repo-Insight", layout="wide")

@st.cache_resource
def get_db_graph():
    db = falkordb.FalkorDB(host=FALKORDB_HOST, port=FALKORDB_PORT)
    return db.select_graph(GRAPH_NAME)

st.title("Repo-Insight: Anti-RAG Coding Agent")
st.markdown("Powered by AMD MI300X & FalkorDB")

if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = ""

st.sidebar.title("Hackathon Demo")
st.sidebar.markdown("Click a button to load a demo prompt:")
DEMO_PROMPTS = [
    "Add a `decorators` field to the FunctionDef dataclass and update all code that creates or reads FunctionDef instances.",
    "Rename `get_connection` to `connect_to_graph` everywhere in the codebase. List every file that needs changing.",
    "Add error handling to `parse_file` so it returns a partial result on syntax errors. What other functions need to handle this new behavior?",
    "Add a `timeout` parameter to `run_ingestion` and propagate it to all database calls it makes."
]
for idx, p in enumerate(DEMO_PROMPTS, 1):
    if st.sidebar.button(f"Demo {idx}: {p[:40]}..."):
        st.session_state.selected_prompt = p

def get_edge_color(types):
    if "CALLS" in types: return "#007bff"
    if "INHERITS_FROM" in types: return "#6f42c1"
    if "IMPORTS" in types: return "#28a745"
    return "#6c757d"

col1, col2 = st.columns([1, 1.2])

with col1:
    st.header("A/B Agent Execution")
    st.markdown("Compare the blind Baseline LLM vs our Graph-Grounded Agent.")
    query = st.text_area("Task for the Agent:", value=st.session_state.selected_prompt, height=120)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_baseline = st.button("Run Baseline (Blind LLM)", type="secondary", use_container_width=True)
    with col_btn2:
        run_agent = st.button("Run Graph-RAG Agent", type="primary", use_container_width=True)
        
    if run_baseline:
        if query:
            with st.spinner("Querying LLM blindly (Baseline)..."):
                client = openai.OpenAI(base_url=SGLANG_BASE_URL, api_key=SGLANG_API_KEY)
                start = time.time()
                try:
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a helpful coding assistant. When asked to make code changes, list every file that needs modification and describe what changes to make. Be thorough — missing a file means the change will break things."},
                            {"role": "user", "content": query},
                        ],
                    )
                    elapsed = time.time() - start
                    st.error(f"### ❌ Baseline Result ({elapsed:.1f}s)\n\n" + response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error: {e}")

    if run_agent:
        if query:
            with st.spinner("Graph-RAG Agent is Architecting and executing..."):
                try:
                    start = time.time()
                    res = run_repo_agent(query, get_db_graph())
                    elapsed = time.time() - start
                    
                    st.success(f"### ✅ Graph-RAG Result ({elapsed:.1f}s | {len(res.get('tool_calls_log', []))} queries)\n\n" + res.get("answer", "No answer provided."))
                    if "diff" in res and res["diff"]:
                        with st.expander("View Unified Diff"):
                            st.code(res["diff"], language="diff")
                    with st.expander("View Agent Tool Log"):
                        st.json(res.get("tool_calls_log", []))
                except Exception as e:
                    st.error(f"Error running agent: {e}")

with col2:
    st.header("Semantic Zoom Visualizer")
    zoom_level = st.selectbox("Zoom Level (The Highway Model)", ["Macro Architecture (Thick Edges)", "Class Architecture (Medium Edges)"])
    
    col_ui1, col_ui2 = st.columns(2)
    with col_ui1:
        min_weight = st.slider("Filter Thin Edges (Min Weight)", min_value=1, max_value=50, value=10)
    with col_ui2:
        max_nodes = st.slider("Max Nodes to Display", min_value=10, max_value=300, value=100)
        
    enable_physics = st.checkbox("Enable Physics (Warning: Heavy CPU usage)", value=False)
    
    graph = get_db_graph()
    nodes = []
    edges = []
    
    if zoom_level == "Macro Architecture (Thick Edges)":
        data = get_macro_architecture(graph)
        
        # Determine node degrees to filter out leaf nodes
        node_degrees = Counter()
        for e in data.get("modules", []):
            if e["weight"] >= min_weight:
                node_degrees[e["source"]] += 1
                node_degrees[e["target"]] += 1
                
        # Only keep top N nodes
        top_nodes = set([n for n, count in node_degrees.most_common(max_nodes)])
        
        modules = set()
        for e in data.get("modules", []):
            if e["weight"] < min_weight: continue
            if e["source"] not in top_nodes or e["target"] not in top_nodes: continue
            
            modules.add(e["source"])
            modules.add(e["target"])
            
            label = " & ".join(e["types"])
            size = min(max(e["weight"] / 2, 1), 10)
            color = get_edge_color(e["types"])
            edges.append(Edge(source=e["source"], target=e["target"], label=label, width=size, color=color))
            
        for m in modules:
            # Scale node size slightly based on degree
            deg = min(node_degrees[m], 15)
            nodes.append(Node(id=m, label=m, color="#FF4B4B", size=15 + deg))
            
    elif zoom_level == "Class Architecture (Medium Edges)":
        mod = st.text_input("Enter Module Name (e.g. your_module):", value="")
        if mod:
            data = get_class_architecture(mod, graph)
            
            node_degrees = Counter()
            for e in data.get("class_edges", []):
                if e["weight"] >= min_weight:
                    node_degrees[e["source"]] += 1
                    node_degrees[e["target"]] += 1
                    
            top_nodes = set([n for n, count in node_degrees.most_common(max_nodes)])
            
            classes = set()
            for e in data.get("class_edges", []):
                if e["weight"] < min_weight: continue
                if e["source"] not in top_nodes or e["target"] not in top_nodes: continue
                
                classes.add(e["source"])
                classes.add(e["target"])
                
                label = " & ".join(e["types"])
                size = min(max(e["weight"] / 2, 1), 10)
                color = get_edge_color(e["types"])
                edges.append(Edge(source=e["source"], target=e["target"], label=label, width=size, color=color))
                
            for c in classes:
                deg = min(node_degrees[c], 15)
                nodes.append(Node(id=c, label=c, color="#4B4BFF", size=15 + deg))
        else:
            st.info("Enter a module name to see its internal class structure.")
            
    config = Config(width=800, 
                    height=650, 
                    directed=True, 
                    nodeHighlightBehavior=True, 
                    highlightColor="#F7A7A6", 
                    collapsible=False,
                    physics=enable_physics)
                    
    if nodes:
        agraph(nodes=nodes, edges=edges, config=config)
    elif zoom_level == "Macro Architecture (Thick Edges)":
        st.info("Graph is empty or all edges filtered out. Please lower the Min Weight.")

# app.py
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
import falkordb
from tools import get_macro_architecture, get_class_architecture
from config import FALKORDB_HOST, FALKORDB_PORT, GRAPH_NAME
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

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Agent Chat")
    st.markdown("The Agent uses a Two-Phase loop: **Map Before You Drive**.")
    query = st.text_area("Task for the Agent:", value=st.session_state.selected_prompt, height=150)
    if st.button("Execute Task"):
        if query:
            with st.spinner("Agent is Architecting and executing..."):
                try:
                    res = run_repo_agent(query, get_db_graph())
                    st.markdown("### Change Set")
                    st.write(res.get("answer", "No answer provided."))
                    with st.expander("View Agent Tool Log"):
                        st.json(res.get("tool_calls_log", []))
                except Exception as e:
                    st.error(f"Error running agent: {e}")

with col2:
    st.header("Semantic Zoom Visualizer")
    zoom_level = st.selectbox("Zoom Level (The Highway Model)", ["Macro Architecture (Thick Edges)", "Class Architecture (Medium Edges)"])
    min_weight = st.slider("Filter Thin Edges (Min Weight)", min_value=1, max_value=20, value=3)
    
    graph = get_db_graph()
    nodes = []
    edges = []
    
    if zoom_level == "Macro Architecture (Thick Edges)":
        data = get_macro_architecture(graph)
        modules = set()
        for e in data.get("modules", []):
            if e["weight"] < min_weight: continue
            modules.add(e["source"])
            modules.add(e["target"])
            
            # Make edge label out of types
            label = " & ".join(e["types"])
            # Scale edge size for UI
            size = min(max(e["weight"] / 2, 1), 10)
            color = get_edge_color(e["types"])
            
            edges.append(Edge(source=e["source"], target=e["target"], label=label, width=size, color=color))
            
        for m in modules:
            nodes.append(Node(id=m, label=m, color="#FF4B4B", size=25))
            
    elif zoom_level == "Class Architecture (Medium Edges)":
        mod = st.text_input("Enter Module Name (e.g. your_module):", value="")
        if mod:
            data = get_class_architecture(mod, graph)
            classes = set()
            for e in data.get("class_edges", []):
                if e["weight"] < min_weight: continue
                classes.add(e["source"])
                classes.add(e["target"])
                
                label = " & ".join(e["types"])
                size = min(max(e["weight"] / 2, 1), 10)
                color = get_edge_color(e["types"])
                edges.append(Edge(source=e["source"], target=e["target"], label=label, width=size, color=color))
                
            for c in classes:
                nodes.append(Node(id=c, label=c, color="#4B4BFF", size=25))
        else:
            st.info("Enter a module name to see its internal class structure.")
            
    config = Config(width=800, 
                    height=600, 
                    directed=True, 
                    nodeHighlightBehavior=True, 
                    highlightColor="#F7A7A6", 
                    collapsible=False,
                    physics=True)
                    
    if nodes:
        agraph(nodes=nodes, edges=edges, config=config)
    elif zoom_level == "Macro Architecture (Thick Edges)":
        st.info("Graph is empty. Please run the ingestion script first.")

"""
MiniEmbed - Interactive Demo
================================
Explore the embedding model's capabilities through a Streamlit dashboard.

Features:
  - Pairwise text similarity (cosine distance)
  - Semantic document search with ranked results
  - Unsupervised text clustering via K-Means
  - Raw embedding vector inspection and visualization
  - Bulk CSV-to-CSV record matching

Run: streamlit run demo.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import io

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference import EmbeddingInference, EmbeddingModelManager

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MiniEmbed Demo",
    page_icon="M",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .result-box {
        background: rgba(100, 100, 100, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: inherit;
    }
    .high-score { border-left: 4px solid #28a745; background: rgba(40, 167, 69, 0.1); }
    .medium-score { border-left: 4px solid #ffc107; background: rgba(255, 193, 7, 0.1); }
    .low-score { border-left: 4px solid #dc3545; background: rgba(220, 53, 69, 0.1); }
    .score-text { font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model(model_name):
    """Load the embedding model from disk."""
    model_dir = f"models/{model_name}"
    if model_name == "Legacy (model/)":
        model_dir = "model"
    return EmbeddingInference.from_pretrained(model_dir)


# Header
st.markdown('<h1 class="main-header">MiniEmbed Demo</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore semantic similarity, search, clustering, and bulk matching</p>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Selection
# -----------------------------------------------------------------------------
available_models = EmbeddingModelManager.list_models()
if os.path.exists("model/model.pt"):
    available_models.append("Legacy (model/)")

if not available_models:
    st.error("No models found. Train a model first or place weights in models/mini/model.pt.")
    st.info("Models should be located in the `models/` directory (e.g., `models/mini/`).")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "Select Model", 
    available_models,
    index=0,
    help="Select which trained model to load for inference."
)

model = load_model(selected_model_name)

if model is None:
    st.error("Model not found. Please train the model first.")
    st.stop()

# Model info
with st.expander("Model Info", expanded=False):
    st.markdown("""
    This panel shows the architecture of the currently loaded model.
    - **Embedding Dim**: The size of each output vector (higher = more expressive).
    - **Layers**: Number of Transformer encoder layers stacked in the model.
    - **Vocab Size**: Total number of unique tokens the model can recognize.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Embedding Dim", model.model.d_model)
    with col2:
        st.metric("Layers", len(model.model.layers))
    with col3:
        st.metric("Vocab Size", len(model.tokenizer.word_to_id))

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Similarity", 
    "Semantic Search", 
    "Clustering",
    "Encode Text",
    "CSV Matcher"
])

# ============================================================================
# TAB 1: SIMILARITY
# ============================================================================

with tab1:
    st.markdown("### Pairwise Text Similarity")
    st.markdown("""
    Enter two texts to compute their **cosine similarity** (range: 0 to 1).
    The model encodes each text into a 256-dimensional vector and measures
    the angular distance between them. A score close to 1.0 means the texts
    are semantically equivalent; a score near 0.0 means they are unrelated.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area(
            "Text 1",
            "Machine learning is a branch of artificial intelligence",
            height=100,
            key="sim_text1"
        )
    
    with col2:
        text2 = st.text_area(
            "Text 2", 
            "AI systems can learn patterns from data",
            height=100,
            key="sim_text2"
        )
    
    if st.button("Compute Similarity", type="primary", key="sim_btn"):
        if text1 and text2:
            with st.spinner("Computing..."):
                similarity = model.similarity(text1, text2)
            
            if similarity > 0.7:
                color = "#28a745"
                label = "Very Similar"
            elif similarity > 0.4:
                color = "#ffc107"
                label = "Somewhat Similar"
            else:
                color = "#dc3545"
                label = "Not Similar"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; font-weight: bold; color: {color};">
                    {similarity:.3f}
                </div>
                <div style="font-size: 1.2rem; color: {color};">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Example pairs
    st.markdown("---")
    st.markdown("#### Example Pairs")
    st.markdown("These pairs demonstrate how the model distinguishes related from unrelated content:")
    
    examples = [
        ("Python is a programming language", "Java is used for software development"),
        ("The cat sat on the mat", "A feline rested on the rug"),
        ("Machine learning is fascinating", "I love eating pizza"),
    ]
    
    for t1, t2 in examples:
        similarity = model.similarity(t1, t2)
        
        if similarity > 0.5:
            css_class = "high-score"
        elif similarity > 0.3:
            css_class = "medium-score"
        else:
            css_class = "low-score"
        
        st.markdown(f"""
        <div class="result-box {css_class}">
            <strong>{similarity:.3f}</strong> | "{t1}" vs "{t2}"
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: SEMANTIC SEARCH
# ============================================================================

with tab2:
    st.markdown("### Semantic Document Search")
    st.markdown("""
    Enter a natural-language query. The model encodes your query and all
    documents into the same vector space, then ranks documents by cosine
    similarity. This finds **meaning-based** matches, not just keyword overlap.
    """)
    
    default_docs = """Python is a high-level programming language
Machine learning algorithms learn patterns from data
The weather today is sunny and warm
Neural networks are inspired by the human brain
JavaScript is used for web development
Deep learning has transformed computer vision
Cats are popular pets around the world
TensorFlow and PyTorch are ML frameworks
The stock market had a volatile day
Natural language processing understands text"""
    
    query = st.text_input(
        "Search Query",
        "How do AI systems learn from examples?",
        key="search_query"
    )
    
    documents_text = st.text_area(
        "Documents (one per line)",
        default_docs,
        height=200,
        key="search_docs"
    )
    
    top_k = st.slider("Number of results", 1, 10, 5, key="search_topk")
    
    if st.button("Search", type="primary", key="search_btn"):
        documents = [d.strip() for d in documents_text.split('\n') if d.strip()]
        
        if query and documents:
            with st.spinner("Searching..."):
                results = model.search(query, documents, top_k=top_k)
            
            st.markdown("### Results")
            st.markdown("Documents ranked by semantic relevance to your query:")
            
            for r in results:
                score = r['score']
                if score > 0.6:
                    indicator = "[HIGH]"
                    css_class = "high-score"
                elif score > 0.4:
                    indicator = "[MED]"
                    css_class = "medium-score"
                else:
                    indicator = "[LOW]"
                    css_class = "low-score"
                
                st.markdown(f"""
                <div class="result-box {css_class}">
                    <strong>{indicator} #{r['rank']}</strong> (score: {score:.4f})<br>
                    {r['text']}
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# TAB 3: CLUSTERING
# ============================================================================

with tab3:
    st.markdown("### Unsupervised Text Clustering")
    st.markdown("""
    The model encodes each text into a dense vector. K-Means clustering
    then groups these vectors by proximity in the embedding space.
    Texts that are semantically similar end up in the same cluster,
    even if they share no common words.
    """)
    
    default_cluster_texts = """Python programming language
Machine learning algorithms
Deep learning neural networks
JavaScript web development
Cats and dogs as pets
Pizza and pasta Italian food
Sunny weather today
Rainy day forecast
Stock market trends
Financial news update"""
    
    cluster_texts = st.text_area(
        "Texts to cluster (one per line)",
        default_cluster_texts,
        height=200,
        key="cluster_texts"
    )
    
    n_clusters = st.slider("Number of clusters", 2, 10, 3, key="n_clusters")
    
    if st.button("Run Clustering", type="primary", key="cluster_btn"):
        texts = [t.strip() for t in cluster_texts.split('\n') if t.strip()]
        
        if len(texts) >= n_clusters:
            with st.spinner("Clustering..."):
                result = model.cluster_texts(texts, n_clusters=n_clusters)
            
            st.markdown("### Cluster Assignments")
            st.markdown("Each group contains texts that the model considers semantically related:")
            
            colors = ["#667eea", "#28a745", "#ffc107", "#dc3545", "#17a2b8", 
                     "#6f42c1", "#fd7e14", "#20c997", "#e83e8c", "#6c757d"]
            
            for cluster_id in sorted(result['texts_by_cluster'].keys()):
                cluster_texts_list = result['texts_by_cluster'][cluster_id]
                color = colors[cluster_id % len(colors)]
                
                st.markdown(f"""
                <div style="background: {color}15; border-left: 4px solid {color}; 
                            padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                    <strong style="color: {color};">Cluster {cluster_id + 1}</strong>
                    ({len(cluster_texts_list)} texts)
                </div>
                """, unsafe_allow_html=True)
                
                for text in cluster_texts_list:
                    st.markdown(f"  - {text}")
        else:
            st.warning(f"Need at least {n_clusters} texts to create {n_clusters} clusters.")

# ============================================================================
# TAB 4: ENCODE TEXT
# ============================================================================

with tab4:
    st.markdown("### Raw Embedding Inspector")
    st.markdown("""
    Convert any text into its dense vector representation. The output is a
    256-dimensional float vector that is **L2-normalized** (unit length = 1.0).
    This is the same representation used internally for similarity and search.
    """)
    
    encode_text = st.text_area(
        "Text to encode",
        "Machine learning is a fascinating field of study.",
        height=100,
        key="encode_text"
    )
    
    if st.button("Encode", type="primary", key="encode_btn"):
        if encode_text:
            with st.spinner("Encoding..."):
                embedding = model.encode(encode_text)
            
            st.markdown("### Embedding Vector")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Dimensions", embedding.shape[1])
            with col2:
                st.metric("L2 Norm", f"{np.linalg.norm(embedding[0]):.4f}")
            with col3:
                st.metric("Mean Value", f"{embedding[0].mean():.4f}")
            
            st.markdown("#### First 20 values:")
            st.code(str(embedding[0][:20].round(4).tolist()))
            
            st.markdown("#### Value Distribution")
            st.markdown("A well-trained model produces a roughly Gaussian distribution centered near zero:")
            import plotly.express as px
            fig = px.histogram(
                x=embedding[0], 
                nbins=50,
                title="Embedding Value Distribution",
                labels={'x': 'Value', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

# ============================================================================
# TAB 5: CSV MATCHER
# ============================================================================

with tab5:
    st.markdown("### Bulk CSV Record Matcher")
    st.markdown("""
    Upload two CSV files and match rows across them using semantic similarity.
    This is useful for:
    - **Product deduplication** across e-commerce platforms
    - **Record linkage** between databases with inconsistent naming
    - **Cross-platform mapping** (e.g., matching supplier catalogs to your inventory)

    The model encodes the selected text column from each CSV, then ranks
    every row in CSV 2 against each row in CSV 1 by cosine similarity.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Upload CSV 1 (Queries)")
        file1 = st.file_uploader("Upload primary CSV", type=['csv'], key="csv_file_1")
    
    with col2:
        st.markdown("#### Upload CSV 2 (Knowledge Base)")
        file2 = st.file_uploader("Upload secondary CSV", type=['csv'], key="csv_file_2")

    if file1 and file2:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            col1_name = st.selectbox("Select column to match from CSV 1", df1.columns, key="col1_sel")
        
        with col_m2:
            col2_name = st.selectbox("Select column to search in CSV 2", df2.columns, key="col2_sel")
            
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            top_n_candidates = st.slider("Step 1: Top candidates to fetch", 1, 50, 10, help="Initial semantic search depth")
        with col_p2:
            top_m_final = st.slider("Step 2: Top matches to keep", 1, 10, 3, help="Final number of matches per row")

        if st.button("Start Bulk Matching", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            queries = df1[col1_name].fillna("").astype(str).tolist()
            corpus = df2[col2_name].fillna("").astype(str).tolist()
            
            status_text.text("Encoding search corpus (CSV 2)...")
            corpus_embs = model.encode(corpus, batch_size=128)
            progress_bar.progress(20)
            
            status_text.text("Encoding queries (CSV 1)...")
            query_embs = model.encode(queries, batch_size=128)
            progress_bar.progress(50)
            
            status_text.text("Computing similarities and mapping...")
            similarities = np.dot(query_embs, corpus_embs.T)
            progress_bar.progress(80)
            
            all_results = []
            for i in range(len(queries)):
                row_scores = similarities[i]
                top_indices = np.argsort(row_scores)[::-1][:top_m_final]
                
                res_row = df1.iloc[i].to_dict()
                for rank, idx in enumerate(top_indices, 1):
                    res_row[f'Match_{rank}_{col2_name}'] = corpus[idx]
                    res_row[f'Match_{rank}_Score'] = round(float(row_scores[idx]), 4)
                all_results.append(res_row)
            
            res_df = pd.DataFrame(all_results)
            
            progress_bar.progress(100)
            status_text.text("Matching complete.")
            
            st.markdown("### Results Preview")
            st.dataframe(res_df.head(50), width="stretch")
            
            output = io.StringIO()
            res_df.to_csv(output, index=False)
            csv_string = output.getvalue()
            
            st.download_button(
                label="Download Full Results CSV",
                data=csv_string,
                file_name="semantic_matching_results.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload both CSV files to begin matching.")


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <strong>MiniEmbed</strong> | Lightweight Text Embeddings | 
    <a href="https://github.com/bhandarisuraz/miniembed">GitHub</a>
</div>
""", unsafe_allow_html=True)

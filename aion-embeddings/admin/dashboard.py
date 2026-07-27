# admin/dashboard.py

import streamlit as st
import pandas as pd
from storage.database import get_connection, get_state, set_state
from core.pattern_learner import PatternLearner

st.set_page_config(page_title="AION Admin Dashboard", layout="wide")

st.title("AION Embeddings Admin Dashboard")

tab1, tab2, tab3 = st.tabs(["System State", "Learned Patterns", "Training Logs"])

with tab1:
    st.header("System State")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Questions Seen", get_state("total_questions_seen") or "0")
        st.metric("Total Training Runs", get_state("total_training_runs") or "0")
        
    with col2:
        st.metric("Current Model Version", get_state("current_model_version") or "v0")
        st.metric("Last Training Time", get_state("last_training_time") or "Never")
        
    with col3:
        frozen = get_state("model_frozen") == "true"
        if st.button("Freeze Model" if not frozen else "Unfreeze Model"):
            set_state("model_frozen", "false" if frozen else "true")
            st.rerun()
            
    st.subheader("Recent Uploads")
    with get_connection() as conn:
        uploads = pd.read_sql_query("SELECT filename, subject, file_size, question_count, processed, created_at FROM uploaded_files ORDER BY created_at DESC LIMIT 10", conn)
        st.dataframe(uploads)

with tab2:
    st.header("Learned Patterns")
    
    learner = PatternLearner()
    stats = learner.get_statistics()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Question Templates")
        if stats["top_patterns"]:
            df = pd.DataFrame(stats["top_patterns"])
            st.dataframe(df)
        else:
            st.info("No patterns learned yet.")
            
    with col2:
        st.subheader("Bloom Taxonomy Distribution")
        if stats["bloom_distribution"]:
            st.bar_chart(pd.Series(stats["bloom_distribution"]))
        else:
            st.info("No bloom distribution yet.")
            
        st.subheader("Question Type Distribution")
        if stats["type_distribution"]:
            st.bar_chart(pd.Series(stats["type_distribution"]))
            
with tab3:
    st.header("Training Logs")
    with get_connection() as conn:
        logs = pd.read_sql_query("SELECT version, status, pairs_used, epochs, eval_score, duration_seconds, started_at FROM training_log ORDER BY started_at DESC LIMIT 20", conn)
        st.dataframe(logs)

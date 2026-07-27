import streamlit as st
import requests
import json
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AION VTU Generator", layout="wide")

st.title("🧠 AION VTU Paper Generator")

tab1, tab2 = st.tabs(["Generate Paper", "HOD Review Dashboard"])

with tab1:
    st.header("Generate New Paper")
    with st.form("gen_form"):
        subject_name = st.text_input("Subject Name", value="Advanced Databases")
        subject_code = st.text_input("Subject Code", value="18CS51")
        modules = st.multiselect("Modules to Cover", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        
        submitted = st.form_submit_button("Generate Paper")
        if submitted:
            with st.spinner("Initializing generation pipeline..."):
                payload = {
                    "subject": subject_name,
                    "subject_code": subject_code,
                    "modules_to_cover": modules
                }
                try:
                    res = requests.post(f"{API_URL}/generate", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Generation started! Run ID: {data['run_id']}")
                        st.session_state['last_run_id'] = data['run_id']
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

with tab2:
    st.header("HOD Review Dashboard")
    st.write("In production, this would list all pending runs. For now, enter a Run ID.")
    
    run_id_input = st.text_input("Run ID", value=st.session_state.get('last_run_id', ''))
    
    if st.button("Fetch Paper") and run_id_input:
        try:
            res = requests.get(f"{API_URL}/paper/{run_id_input}")
            if res.status_code == 200:
                data = res.json()
                status = data.get("status")
                
                st.subheader(f"Status: {status.upper()}")
                
                if data.get("generation_errors"):
                    st.error("Generation Errors:")
                    st.write(data["generation_errors"])
                
                if data.get("validation_errors"):
                    st.warning("Validation Errors:")
                    st.write(data["validation_errors"])
                
                draft = data.get("draft_paper")
                if draft:
                    st.write("### Draft Paper")
                    st.json(draft)
                    
                    if status == "pending_review":
                        st.write("### Submit Review")
                        verdict = st.selectbox("Verdict", ["approve", "revise", "reject"])
                        feedback = st.text_area("Feedback (required for revise/reject)")
                        
                        if st.button("Submit Review"):
                            rev_payload = {"verdict": verdict, "feedback": feedback}
                            rev_res = requests.post(f"{API_URL}/review/{run_id_input}", json=rev_payload)
                            if rev_res.status_code == 200:
                                st.success("Review processed!")
                                st.rerun()
                            else:
                                st.error(f"Failed to submit: {rev_res.text}")
            else:
                st.error("Run ID not found or error occurred.")
        except Exception as e:
            st.error(f"API Connection Error: {e}")

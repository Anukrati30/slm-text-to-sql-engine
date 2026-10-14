import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Enterprise Text-to-SQL Platform", 
    page_icon="🧠", 
    layout="wide"
)

st.title("🧠 Enterprise Text-to-SQL Analytics Platform")
st.markdown("Convert natural language questions into executable queries, run them against a secure database sandbox, and inspect real-time results.")

# Sidebar scenario presets
st.sidebar.header("📁 Preset Scenarios")
preset = st.sidebar.selectbox(
    "Choose a schema template", 
    ["Custom Input", "Employees & Departments", "Users & Orders", "Products & Inventory"]
)

default_evidence = "TABLE employees (id INT, name VARCHAR, department VARCHAR, salary FLOAT); TABLE departments (id INT, name VARCHAR, location VARCHAR)"
default_question = "What is the average salary of employees working in the 'Engineering' department?"

if preset == "Users & Orders":
    default_evidence = "TABLE users (id INT, signup_date DATE, country VARCHAR); TABLE orders (id INT, user_id INT, amount FLOAT, status VARCHAR)"
    default_question = "Find the total revenue from completed orders for users in 'USA' who signed up in 2025."
elif preset == "Products & Inventory":
    default_evidence = "TABLE products (id INT, name VARCHAR, stock INT, price FLOAT)"
    default_question = "List the top 5 most expensive products that currently have stock greater than zero."
elif preset == "Custom Input":
    default_evidence = ""
    default_question = ""

# Two-column layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Natural Language Input")
    with st.form("sql_form"):
        evidence = st.text_area("Database Schema & Evidence", value=default_evidence, height=140)
        question = st.text_input("Natural Language Question", value=default_question)
        submitted = st.form_submit_button("Generate & Execute SQL 🚀", use_container_width=True)

with col2:
    st.subheader("⚡ Execution Sandbox & Results")
    if submitted:
        if not question or not evidence:
            st.warning("⚠️ Please provide both schema evidence and a question.")
        else:
            with st.spinner("🤖 Fine-tuned SLM generating code & querying sandbox..."):
                try:
                    payload = {"question": question, "evidence": evidence}
                    response = requests.post("http://localhost:8000/generate_sql", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✨ SQL Generated Successfully!")
                        st.code(data.get("sql_query"), language="sql")
                        
                        status = data.get("execution_status")
                        if status == "success":
                            st.markdown("### 📊 Database Output Table")
                            cols = data.get("columns", [])
                            rows = data.get("rows", [])
                            if cols and rows:
                                result_df = pd.DataFrame(rows, columns=cols)
                                st.dataframe(result_df, use_container_width=True)
                            else:
                                st.info("Query executed successfully, but returned 0 rows.")
                        else:
                            st.error(f"❌ SQL Execution Error: {data.get('error_message')}")
                    else:
                        st.error(f"❌ API Error: {response.text}")
                except Exception as e:
                    st.error(f"🔌 Connection Error: Could not connect to backend ({e})")
    else:
        st.info("👈 Fill out your schema and question on the left, then click **Generate & Execute SQL**.")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import sqlite3
import pandas as pd
from unsloth import FastLanguageModel

app = FastAPI(
    title="Enterprise Text-to-SQL Platform",
    description="Production-grade serving with live SQLite execution sandbox",
    version="3.0"
)

print("🚀 Loading fine-tuned model for inference...")
max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="bird_sql_model",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

class SQLRequest(BaseModel):
    question: str
    evidence: str = "None provided"

class SQLResponse(BaseModel):
    sql_query: str
    execution_status: str
    columns: list = []
    rows: list = []
    error_message: str = None

bird_prompt = """Below is a natural language question about a database, paired with schema evidence. Write the correct SQL query to answer the question.

### Question:
{question}

### Schema Evidence:
{evidence}

### SQL:
{sql}"""

def setup_mock_database(evidence: str):
    """Creates a temporary in-memory SQLite database and populates mock tables based on schema evidence."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    # Create default tables depending on keywords in evidence to make testing smooth
    try:
        if "employees" in evidence.lower():
            cursor.execute("CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary FLOAT);")
            cursor.execute("INSERT INTO employees VALUES (1, 'Alice', 'Engineering', 95000);")
            cursor.execute("INSERT INTO employees VALUES (2, 'Bob', 'Marketing', 65000);")
            cursor.execute("INSERT INTO employees VALUES (3, 'Charlie', 'Engineering', 110000);")
        
        if "departments" in evidence.lower():
            cursor.execute("CREATE TABLE departments (id INT, name VARCHAR, location VARCHAR);")
            cursor.execute("INSERT INTO departments VALUES (1, 'Engineering', 'New York');")
            cursor.execute("INSERT INTO departments VALUES (2, 'Marketing', 'San Francisco');")
            
        if "users" in evidence.lower():
            cursor.execute("CREATE TABLE users (id INT, signup_date DATE, country VARCHAR);")
            cursor.execute("INSERT INTO users VALUES (1, '2025-01-15', 'USA');")
            cursor.execute("INSERT INTO users VALUES (2, '2025-03-22', 'UK');")
            
        if "orders" in evidence.lower():
            cursor.execute("CREATE TABLE orders (id INT, user_id INT, amount FLOAT, status VARCHAR);")
            cursor.execute("INSERT INTO orders VALUES (101, 1, 250.0, 'completed');")
            cursor.execute("INSERT INTO orders VALUES (102, 2, 150.0, 'pending');")
            
        if "products" in evidence.lower():
            cursor.execute("CREATE TABLE products (id INT, name VARCHAR, stock INT, price FLOAT);")
            cursor.execute("INSERT INTO products VALUES (1, 'Laptop', 12, 1200.0);")
            cursor.execute("INSERT INTO products VALUES (2, 'Mouse', 45, 25.0);")
            cursor.execute("INSERT INTO products VALUES (3, 'Monitor', 0, 300.0);")
            
        conn.commit()
    except Exception as e:
        pass
        
    return conn

@app.post("/generate_sql", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    try:
        prompt = bird_prompt.format(
            question=request.question,
            evidence=request.evidence,
            sql=""
        )
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=True,
            temperature=0.1,
            do_sample=False
        )
        
        decoded = tokenizer.batch_decode(outputs)[0]
        
        if "### SQL:" in decoded:
            sql_query = decoded.split("### SQL:")[1].strip()
            sql_query = sql_query.replace(tokenizer.eos_token, "").strip()
        else:
            sql_query = decoded
            
        # Execute query in our SQLite sandbox
        conn = setup_mock_database(request.evidence)
        try:
            df = pd.read_sql_query(sql_query, conn)
            columns = df.columns.tolist()
            rows = df.values.tolist()
            conn.close()
            return SQLResponse(
                sql_query=sql_query,
                execution_status="success",
                columns=columns,
                rows=rows
            )
        except Exception as db_err:
            conn.close()
            return SQLResponse(
                sql_query=sql_query,
                execution_status="error",
                error_message=str(db_err)
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

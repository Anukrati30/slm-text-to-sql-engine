# 🧠 Enterprise Text-to-SQL Small Language Model (SLM) & Analytics Platform

A production-grade, end-to-end AI system that fine-tunes an open-source foundational model for domain-specific code generation, served via an asynchronous FastAPI backend and an interactive Streamlit analytics dashboard.

---

## 🏗️ Architecture Overview

```text
┌─────────────────┐      HTTP POST      ┌──────────────────┐      QLoRA / vLLM     ┌────────────────────┐
│  Streamlit UI   │ ──────────────────► │  FastAPI Backend │ ────────────────────► │ Fine-Tuned Llama 3 │
│  (Frontend)     │                     │  (Pydantic/Async)│                       │ (8B via Unsloth)   │
└─────────────────┘                     └──────────────────┘                       └────────────────────┘
         │                                        │                                           │
         └────────────────────────────────────────┼───────────────────────────────────────────┘
                                                  ▼
                                      [ SQLite Execution Sandbox ]
                                      (Executes & Validates Query)

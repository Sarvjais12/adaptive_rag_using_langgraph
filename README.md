# 🧠 Adaptive RAG Assistant (LangGraph + Groq)

### **Dynamic Retrieval-Augmented Generation with Stateful Memory**
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Sarvjais12/adaptive-rag-assistant)

This project is an advanced RAG implementation developed during my 6th semester to solve the limitations of standard, static retrieval systems. It uses **LangGraph** to orchestrate a multi-strategy workflow that adapts to query complexity in real-time.

---

## 🚀 Key Features

* **Adaptive Strategy Selection**: Uses a scoring algorithm to switch between **Light**, **Standard**, and **Deep** retrieval paths based on user intent.
* **Stateful Multi-Turn Conversation**: Solved the "LLM Amnesia" problem by injecting `chat_history` into the LangGraph state, allowing the model to remember context across multiple questions.
* **Hallucination Guardrails**: Implemented strict "Sandwich Constraints" and negative prompting to ensure the model only answers based on the uploaded document.
* **High-Speed Inference**: Integrated **Groq LPU** (Llama-3.1-8b) for near-instantaneous responses.
* **Hybrid Vector Search**: Combines semantic similarity search with **Max Marginal Relevance (MMR)** to ensure retrieved context is both relevant and diverse.

---

## 🏗️ How it Works (The Architecture)

The system follows a state-machine workflow defined in LangGraph:

1.  **Complexity Analysis**: The query is scored based on length and specific keywords (e.g., "compare," "explain").
2.  **Retrieval Node**: 
    * **Light**: Simple similarity search ($k = 4$).
    * **Standard**: Diversified MMR search ($k = 6$).
    * **Deep**: Expanded query logic to pull broader context ($k = 10$).
3.  **Generation Node**: Context and chat history are merged. The temperature is locked at 0.1 for maximum factual accuracy.
4.  **Validation Node**: Ensures the output meets the required length and quality before returning to the UI.

---

## 🛠️ Tech Stack

* **Logic**: LangGraph, LangChain
* **LLM**: Meta Llama-3.1-8b (via Groq Cloud)
* **Vector Database**: FAISS
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
* **Interface**: Gradio

---

## 📊 Sample Use Case: Environmental Ethics
When tested with an 11-page philosophical draft:
* **Simple Query**: "What is anthropocentrism?" -> Triggers **Light Strategy**.
* **Complex Query**: "Explain the fundamental differences between strong anthropocentrism and ecocentric holism." -> Triggers **Deep Strategy** for a more nuanced, multi-chunk analysis.

---

## 🔗 Live Demo
Try it out on Hugging Face Spaces: [Adaptive RAG Assistant](https://huggingface.co/spaces/Sarvjais12/adaptive-rag-assistant)

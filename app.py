import gradio as gr
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
from groq import Groq

# ============ DEFAULT KNOWLEDGE BASE ============
DEFAULT_KNOWLEDGE = """LangGraph is a library for building stateful, multi-actor applications with LLMs.
It provides stateful workflows, cyclic graphs, persistence, and streaming.

RAG: Retrieval-Augmented Generation enhances LLMs with external knowledge retrieval.
Adaptive RAG: Dynamically adjusts retrieval strategy based on query complexity.
Stateful Workflows: LangGraph maintains state across multiple steps using checkpointing."""

class AdaptiveRAG:
    def __init__(self, vector_db, embeddings):
        self.db = vector_db
        self.embeddings = embeddings
        self.query_history = []
        
    def analyze_query_complexity(self, query: str):
        words = query.split()
        complex_indicators = ['how', 'why', 'explain', 'compare', 'difference', 'architecture', 'detail']
        has_complex = any(ind in query.lower() for ind in complex_indicators)
        
        score = 0.3 if len(words) > 8 else 0
        score += 0.4 if has_complex else 0
        
        if score < 0.3:
            return {"k": 4, "strategy": "light"}
        elif score < 0.6:
            return {"k": 6, "strategy": "standard"}
        else:
            return {"k": 10, "strategy": "deep"}
    
    def retrieve(self, query: str):
        config = self.analyze_query_complexity(query)
        strategy = config["strategy"]
        
        if strategy == "light":
            results = self.db.similarity_search(query, k=config["k"])
        elif strategy == "standard":
            results = self.db.max_marginal_relevance_search(query, k=config["k"], fetch_k=20)
        else:
            initial = self.db.similarity_search(query, k=30)
            expanded = f"{query} " + " ".join([d.page_content[:200] for d in initial[:3]])
            results = self.db.similarity_search(expanded, k=config["k"])
            
        self.query_history.append({"query": query, "strategy": strategy, "count": len(results)})
        return results, strategy

class GraphState(TypedDict):
    question: str
    chat_history: List[Dict[str, Any]]
    documents: List
    generation: str
    strategy: str
    steps: Annotated[List[str], operator.add]

class AdaptiveRAGWorkflow:
    def __init__(self, rag_system, llm: Optional = None):
        self.rag = rag_system
        self.llm = llm
        self.workflow = self._create_workflow()
        
    def _create_workflow(self):
        def retrieve(state: GraphState):
            docs, strategy = self.rag.retrieve(state["question"])
            # Pass the existing history through the node
            return {"documents": docs, "strategy": strategy, "question": state["question"], "chat_history": state.get("chat_history", []), "steps": ["retrieve"]}
        
        def generate(state: GraphState):
            question = state["question"]
            docs = state["documents"]
            history = state.get("chat_history", [])
            
            context = "\n\n---\n\n".join([d.page_content for d in docs])
            
            if self.llm:
                messages = [
                    {"role": "system", "content": f"You are an expert AI research assistant. Answer the user's question accurately and thoroughly based ONLY on the provided context. If the answer is not in the context, state that clearly.\n\nContext:\n{context}"}
                ]
                
                # Inject the conversation memory
                for msg in history:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    
                messages.append({
    "role": "user", 
    "content": f"Question: {question}\n\nCRITICAL INSTRUCTION: You are strictly forbidden from using outside knowledge. If the provided context does not contain the exact answer, you MUST reply EXACTLY with: 'I cannot answer this based on the provided document.'\n\nDetailed Answer:"
})

                try:
                    completion = self.llm.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                    generation = completion.choices[0].message.content.strip()
                    if "Detailed Answer:" in generation:
                        generation = generation.split("Detailed Answer:")[-1].strip()
                except Exception as e:
                    generation = f"⚠️ **API ERROR:** {str(e)}\n\n---\n\n" + self._fallback_answer(question, context, docs, state["strategy"])
            else:
                generation = self._fallback_answer(question, context, docs, state["strategy"])
            
            return {"documents": docs, "strategy": state["strategy"], "question": question, "chat_history": history, "generation": generation, "steps": ["generate"]}
        
        def validate(state: GraphState):
            return state
        
        workflow = StateGraph(GraphState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("validate", validate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_edge("validate", END)
        
        return workflow.compile()
    
    def _fallback_answer(self, question, context, docs, strategy):
        q = question.lower()
        words_in_q = q.split()
        if any(w in words_in_q for w in ['hi', 'hello', 'hey']):
            return "Hello! I'm your Adaptive RAG Assistant. Upload a document or ask me a question!"
        info = context[:400]
        return f"**Answer based on context:**\n• {info}...\n\n_Retrieved using {strategy} strategy. (Add API key for full AI generation)_"
    
    def run(self, question: str, history: List[Dict[str, Any]]):
        return self.workflow.invoke({
            "question": question,
            "chat_history": history,
            "documents": [],
            "generation": "",
            "strategy": "",
            "steps": []
        })

# ============ GLOBAL VARIABLES ============
adaptive_rag = None
workflow = None
current_llm = None

def build_vector_db(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
        
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings), embeddings

def initialize_system(api_key=None, uploaded_file=None):
    global adaptive_rag, workflow, current_llm
    
    if uploaded_file is not None:
        file_path = uploaded_file.name
        status_msg = f"✅ Initialized with: {os.path.basename(file_path)}"
    else:
        with open('temp_default.txt', 'w', encoding='utf-8') as f:
            f.write(DEFAULT_KNOWLEDGE)
        file_path = 'temp_default.txt'
        status_msg = "✅ Initialized with default knowledge base."
        
    vector_db, embeddings = build_vector_db(file_path)
    
    llm = None
    if api_key and api_key.strip().startswith('gsk_'):
        try:
            llm = Groq(api_key=api_key.strip())
            current_llm = "Llama-3.1 (Groq)"
        except Exception as e:
            current_llm = f"Error: {str(e)[:50]}"
    else:
        current_llm = "None (Fallback Mode)"
    
    adaptive_rag = AdaptiveRAG(vector_db, embeddings)
    workflow = AdaptiveRAGWorkflow(adaptive_rag, llm)
    
    return f"{status_msg} | LLM: {current_llm}"

# ============ UI ============
with gr.Blocks(title="Adaptive RAG Assistant") as demo:
    gr.Markdown("# 🧠 Adaptive RAG Assistant")
    gr.Markdown("### Intelligent Retrieval with PDF Support and Groq LPU Inference")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")
            
            groq_key = gr.Textbox(
                label="Groq API Key (Starts with gsk_)", 
                placeholder="gsk_...",
                type="password"
            )
            
            doc_upload = gr.File(
                label="Upload Knowledge Base (.txt, .pdf)", 
                file_types=[".txt", ".pdf"]
            )
            
            init_btn = gr.Button("🚀 Initialize / Update System", variant="primary")
            status = gr.Textbox(label="Status", value="Not initialized", interactive=False)
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=400, label="Conversation")
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Upload a document and ask a question...",
                    scale=4
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear Chat")
    
    def respond(message, history):
        global workflow
        
        if workflow is None:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⚠️ Please click 'Initialize / Update System' first!"})
            return history
        
        if not message:
            return history
            
        try:
            # Pass the existing history into the workflow BEFORE appending the new message
            result = workflow.run(message, history.copy()) 
            
            strategy = result.get('strategy', 'standard')
            response = result['generation']
            
            history.append({"role": "user", "content": message})
            formatted_response = f"[{strategy.upper()} STRATEGY]\n\n{response}"
            history.append({"role": "assistant", "content": formatted_response})
            
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
            
        return history
    
    init_btn.click(initialize_system, inputs=[groq_key, doc_upload], outputs=[status])
    send_btn.click(respond, inputs=[msg_input, chatbot], outputs=[chatbot]).then(
        lambda: "", outputs=[msg_input]
    )
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == "__main__":
    initialize_system()
    demo.launch()

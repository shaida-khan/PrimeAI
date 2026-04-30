import os
import uuid
import gradio as gr

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_classic.chains import RetrievalQA


MODEL_NAME = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
MAX_FILE_SIZE_MB = 20
MAX_QUESTION_LENGTH = 2000


def check_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def load_document(file):
    try:
        if file is None:
            return None, "Please upload one PDF or Word document first."

        file_path = file.name if hasattr(file, "name") else file
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        file_size = check_file_size(file_path)
        if file_size > MAX_FILE_SIZE_MB:
            return None, f"❌ File is too large: {file_size:.1f} MB. Maximum allowed size is {MAX_FILE_SIZE_MB} MB."

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            return None, "❌ Unsupported file type. Please upload only .pdf or .docx."

        docs = loader.load()
        full_text = "\n\n".join([doc.page_content for doc in docs])

        for doc in docs:
            doc.metadata["source_file"] = file_name

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"primeai_doc_{uuid.uuid4().hex}"
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        return retriever, (
            f"✅ Document loaded successfully\n\n"
            f"📄 File: {file_name}\n"
            f"📦 Size: {file_size:.1f} MB\n"
            f"📚 Sections/pages: {len(docs)}\n\n"
            f"Ready for questions."
        ), full_text

    except Exception as e:
        return None, f"❌ Error loading document: {str(e)}"


def format_sources(source_documents):
    sources = []

    for doc in source_documents:
        source_file = doc.metadata.get("source_file", "Uploaded document")
        page = doc.metadata.get("page", None)

        if page is not None:
            sources.append(f"{source_file}, page {page + 1}")
        else:
            sources.append(source_file)

    unique_sources = []
    for source in sources:
        if source not in unique_sources:
            unique_sources.append(source)

    if not unique_sources:
        return ""

    return "\n\n**Sources used:**\n" + "\n".join([f"- {s}" for s in unique_sources])


def ask_question(message, chat_history, retriever, mode):
    if chat_history is None:
        chat_history = []

    if not message or message.strip() == "":
        return "", chat_history

    if len(message) > MAX_QUESTION_LENGTH:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({
            "role": "assistant",
            "content": f"Your question is too long. Please keep it under {MAX_QUESTION_LENGTH} characters."
        })
        return "", chat_history

    if retriever is None:
        chat_history.append({"role": "user", "content": message})
        chat_history.append({
            "role": "assistant",
            "content": "Please upload one PDF or Word document first."
        })
        return "", chat_history

    
    
    try:
        llm = ChatOllama(model=MODEL_NAME, temperature=0.2)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        if mode == "Resume Analyzer":
            prompt = f"""
You are PrimeAI App, a professional AI resume strategist and ATS optimization expert.

Analyze the uploaded resume deeply and professionally.

Return the answer using this structure:

# PrimeAI Resume Report

## 1. Resume Score
Give a score out of 100 with reason.

## 2. ATS Score
Give an ATS score out of 100.

## 3. Top Strengths
List the best parts of this resume.

## 4. Weaknesses / Gaps
Explain what is hurting this candidate.

## 5. Missing Keywords
List keywords that should be added.

## 6. Best-Fit Jobs
Recommend realistic job titles.

## 7. Rewrite Weak Bullet Points
Rewrite weak resume bullets professionally.

## 8. Final Improvement Plan
Give clear next steps.

Use only the uploaded resume.

User Request:
{message}
"""
        elif mode == "Job Matching":
            prompt = f"""
You are PrimeAI App, a senior AI career strategist.

The user is an experienced Civil Engineer transitioning into Data Science and AI.

IMPORTANT:
Do NOT treat this as a beginner.
Leverage domain expertise.

Return:

1. Top Job Roles (Domain + AI)
   - Focus on roles combining Civil Engineering + Data/AI

2. Domain Advantage
   - Explain how Civil Engineering experience is a competitive edge

3. Targeted Skill Gaps
   - Only list missing skills needed for transition (not generic)

4. High-Value Projects (CRITICAL)
   - Suggest 3 projects combining Civil Engineering + AI
   - These must be portfolio-ready

5. Transition Strategy
   - Fastest path to move into AI/Data roles

6. Resume Repositioning
   - How to rewrite resume for hybrid roles

Avoid generic advice like “take courses” or “practice Kaggle.”
Be specific and practical.

User Request:
{message}
"""
        else:
            prompt = f"""
You are PrimeAI App, a private AI assistant for documents.

Rules:
- Answer only using the uploaded document.
- Be clear, structured, and practical.
- If the answer is not in the document, say:
  "I could not find that in the document."
- Do not invent information.

Question:
{message}
"""

        result = qa.invoke(prompt)
        answer = result["result"]

        sources = format_sources(result.get("source_documents", []))
        final_answer = answer + sources

    except Exception as e:
        final_answer = f"❌ Error while generating answer: {str(e)}"

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": final_answer})

    return "", chat_history


def clear_chat():
    return []


custom_css = """
body {
    background: #0b1220;
}

.gradio-container {
    background: #0b1220 !important;
    color: white !important;
}

#header {
    text-align: center;
    padding: 30px;
    border: 1px solid #334155;
    border-radius: 18px;
    margin-bottom: 20px;
    background: #111827;
}

#header h1 {
    color: #fde68a;
    font-size: 34px;
    margin-bottom: 8px;
}

#header h3 {
    color: white;
    margin-bottom: 8px;
}

#header p {
    color: #e5e7eb;
}

.gr-button {
    font-weight: bold !important;
}

.gradio-container select,
.gradio-container [role="button"],
.gradio-container .wrap,
.gradio-container .dropdown-arrow {
    cursor: pointer !important;
}
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(chat_history):
    file_path = "primeai_report.pdf"

    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("PrimeAI Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    if not chat_history:
        elements.append(Paragraph("No content available.", styles["BodyText"]))
    else:
    # get last assistant message only
        last_bot = None

        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                last_bot = msg["content"]
                break

        if last_bot is None:
            last_bot = "No analysis found."

        if isinstance(last_bot, list):
            last_bot = " ".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in last_bot
            )

        # Split into lines for better formatting
        lines = str(last_bot).split("\n")

        elements.append(Paragraph("<b>Analysis</b>", styles["Heading2"]))
        elements.append(Spacer(1, 10))

    for line in lines:
        line = line.strip()
            
        if not line:
            elements.append(Spacer(1, 8))

        elif ":" in line and len(line) < 60:
            # Treat as section heading
            elements.append(Paragraph(f"<b>{line}</b>", styles["Heading3"]))
            elements.append(Spacer(1, 6))

        else:
            elements.append(Paragraph(line, styles["BodyText"]))

    doc.build(elements)

    return file_path

with gr.Blocks(css=custom_css, title="PrimeAI App") as demo:
    retriever_state = gr.State(value=None)
    document_text_state = gr.State(value="")

    gr.HTML(
        """
        <div id="header">
            <h1>📄 ✨ PrimeAI App</h1>
            <h3>Private Local Intelligence for Documents, Careers & Projects</h3>
            <p>Analyze documents, optimize resumes, and grow your career locally.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Control Panel")

            doc_file = gr.File(
                label="Upload PDF or Word File",
                file_types=[".pdf", ".docx"]
            )

            mode = gr.Dropdown(
                choices=["Document QA", "Resume Analyzer", "Job Matching"],
                value="Document QA",
                label="PrimeAI Mode"
            )

            reload_btn = gr.Button("Reload Document")

            status = gr.Textbox(
                label="Document Status",
                value="No document loaded yet.",
                lines=7
            )

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="PrimeAI Assistant",
                height=520,
            )

            question = gr.Textbox(
                label="Ask PrimeAI",
                placeholder="Example: Analyze my resume or summarize this document professionally",
                lines=2
            )

            with gr.Row():
                ask_btn = gr.Button("Ask")
                clear_btn = gr.Button("Clear Chat")
            
            download_btn = gr.Button("Download Report")
            file_output = gr.File(label="Download PDF")

    doc_file.change(
        fn=load_document,
        inputs=doc_file,
        outputs=[retriever_state, status, document_text_state]
    ).then(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )

    reload_btn.click(
        fn=load_document,
        inputs=doc_file,
        outputs=[retriever_state, status, document_text_state]
    )

    download_btn.click(
        fn=generate_pdf,
        inputs=chatbot,
        outputs=file_output
    )
    ask_btn.click(
        fn=ask_question,
        inputs=[question, chatbot, retriever_state, mode],
        outputs=[question, chatbot]
    )

    question.submit(
        fn=ask_question,
        inputs=[question, chatbot, retriever_state, mode],
        outputs=[question, chatbot]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot
    )

    gr.Markdown(
        "<center>PrimeAI App • Private local AI workspace powered by Ollama + LangChain</center>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864
    )

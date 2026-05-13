import gradio as gr
from pypdf import PdfReader
import docx2txt
from langchain_community.chat_models import ChatOllama

MODEL_NAME = "llama3.2:3b"


def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def ask_ai(prompt):
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2
    )
    response = llm.invoke(prompt)
    return response.content


def extract_text(file):
    if file is None:
        return "No file uploaded."

    filename = file.name

    if filename.endswith(".pdf"):
        text = ""
        reader = PdfReader(filename)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        return text

    elif filename.endswith(".docx"):
        return docx2txt.process(filename)

    return "Unsupported file."


def resume_analyzer(text):
    prompt = f"""
You are PrimeAI, a professional AI Resume Analyzer and ATS optimization expert.

Analyze this resume professionally.

Return:

# Resume Score
Give a realistic score out of 100 with explanation.

# ATS Score
Give ATS compatibility score.

# Top Strengths
List strongest qualities.

# Weaknesses / Gaps
Explain weaknesses honestly.

# Missing Keywords
List important missing keywords.

# Best-Fit Jobs
Suggest realistic job titles.

# Improvement Plan
Give practical resume improvement advice.

Resume:

{text}
"""
    return ask_ai(prompt)


def job_matching(text, job_description):
    prompt = f"""
You are PrimeAI, a professional AI job matching and career advisor.

Compare the uploaded resume with the job description below.

Return:

# Job Match Score
Give a realistic match score out of 100.

# Strong Matches
List resume strengths that match the job.

# Missing Skills / Gaps
List missing or weak skills.

# Keywords to Add
List keywords the resume should include.

# Best-Fit Job Titles
Suggest related job titles.

# Final Recommendation
Explain whether the candidate should apply and what to improve.

Resume:

{text}

Job Description:

{job_description}
"""
    return ask_ai(prompt)


def document_summary(text, question):
    cleaned_text = clean_text(text)

    prompt = f"""
You are PrimeAI, a professional AI document analysis assistant.

Answer the user's question ONLY using information from the uploaded document.

Your answers must be:
- detailed
- professional
- well-structured
- easy to read

Use:
- headings
- bullet points
- numbered lists when helpful

If the user asks for a summary:
- provide a detailed summary
- explain the main topics
- identify important skills, technologies, concepts, or requirements
- mention key names, certifications, organizations, or achievements

If the answer is not found in the document, say:
"I could not find that information in the uploaded document."

Uploaded Document:

{cleaned_text}

User Question:

{question}
"""
    return ask_ai(prompt)


def respond(message, history, file, mode):
    if not message:
        return "Please type a question."

    text = extract_text(file)

    if text == "No file uploaded.":
        return "Please upload a PDF or Word document first."

    if mode == "Resume Analyzer + ATS":
        return resume_analyzer(text)

    if mode == "Job Matching":
        return job_matching(text, message)

    if mode == "Document QA / Summary":
        return document_summary(text, message)

    return "Please select a mode."


def chat_response(message, history, file, mode):

    if history is None:
        history = []

    if not message:
        return "", history

    try:
        answer = respond(message, history, file, mode)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        return "", history

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})

        return "", history


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
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #334155;
    margin-bottom: 18px;
}

#header h1 {
    color: #fde68a;
    font-size: 34px;
    margin-bottom: 8px;
}

#header p {
    color: #e5e7eb;
    font-size: 16px;
}

#sidebar {
    background: #111827;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #334155;
}

#main-area {
    background: #111827;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #334155;
}

.gr-button {
    font-weight: bold !important;
    border-radius: 12px !important;
}
"""


with gr.Blocks(title="PrimeAI") as demo:

    gr.HTML(
        """
        <div id="header">
            <h1>🤖 PrimeAI</h1>
            <p>
            Private AI Workspace — Document QA, Resume Analyzer,
            Job Matching, and ATS
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar"):
            file_input = gr.File(
                label="Upload PDF or Word Document",
                file_types=[".pdf", ".docx"]
            )

            mode_input = gr.Dropdown(
                choices=[
                    "Document QA / Summary",
                    "Resume Analyzer + ATS",
                    "Job Matching"
                ],
                value="Document QA / Summary",
                label="PrimeAI Mode"
            )

            clear_btn = gr.Button("Clear Chat")

        with gr.Column(scale=3, elem_id="main-area"):
            chatbot = gr.Chatbot(label="PrimeAI Chat")

            message_input = gr.Textbox(
                label="Ask PrimeAI",
                placeholder="Ask a question, analyze a resume, or paste a job description...",
                lines=3
            )

            ask_btn = gr.Button("Ask PrimeAI")

    ask_btn.click(
        fn=chat_response,
        inputs=[message_input, chatbot, file_input, mode_input],
        outputs=[message_input, chatbot]
    )

    message_input.submit(
        fn=chat_response,
        inputs=[message_input, chatbot, file_input, mode_input],
        outputs=[message_input, chatbot]
    )

    clear_btn.click(lambda: [], outputs=chatbot)


demo.launch(server_port=7865, css=custom_css)
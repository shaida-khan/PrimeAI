@echo off
cd /d "C:\Users\smbab\OneDrive\Artificial Intelligence\IBM AI Engineer\Project Generative AI Applications with RAG and LangChain"
start http://127.0.0.1:7864
powershell -ExecutionPolicy Bypass -NoExit -Command ".\.venv\Scripts\Activate.ps1; python app.py"
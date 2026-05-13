from pypdf import PdfReader
import docx2txt

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

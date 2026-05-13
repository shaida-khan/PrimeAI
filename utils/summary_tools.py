def generate_summary(uploaded_text):

    import re

    sentences = re.split(r'(?<=[.!?]) +', uploaded_text)

    summary = " ".join(sentences[:6])

    return summary

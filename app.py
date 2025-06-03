from flask import Flask, render_template, request
import os
import fitz  # PyMuPDF
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

app = Flask(__name__)

# Clean and tokenize text
def clean_text(text):
    translator = str.maketrans("", "", string.punctuation)
    words = text.lower().translate(translator).split()
    return set([word for word in words if word not in ENGLISH_STOP_WORDS])

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Calculate similarity and give feedback
def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    similarity = (tfidf * tfidf.T).toarray()[0, 1]

    resume_words = clean_text(resume_text)
    job_words = clean_text(job_text)
    missing_keywords = job_words - resume_words
    top_missing = list(missing_keywords)[:10]  # top 10 only

    feedback = ""
    if similarity < 0.75 and top_missing:
        feedback = "Consider including keywords like: " + ", ".join(top_missing)

    return round(similarity * 100, 2), feedback

# Web interface
@app.route("/", methods=["GET", "POST"])
def index():
    score = None
    feedback = ""
    if request.method == "POST":
        resume_file = request.files["resume"]
        job_text = request.form["job"]

        resume_path = "resume.pdf"
        resume_file.save(resume_path)
        resume_text = extract_text_from_pdf(resume_path)

        score, feedback = calculate_similarity(resume_text, job_text)
        os.remove(resume_path)

    return render_template("index.html", score=score, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)
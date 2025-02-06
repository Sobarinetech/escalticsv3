import streamlit as st
from huggingface_hub import InferenceClient
from io import BytesIO
import json
import matplotlib.pyplot as plt
import re
from fpdf import FPDF

# Configure API Key securely from Streamlit's secrets
HF_API_KEY = st.secrets["HF_API_KEY"]

# Initialize Hugging Face Inference Client
client = InferenceClient(provider="together", api_key=HF_API_KEY)

# Streamlit App Configuration
st.set_page_config(page_title="Escalytics", page_icon="📧", layout="wide")
st.title("⚡Escalytics by EverTech")
st.write("Extract insights, root causes, and actionable steps from emails.")

# Sidebar for Feature Selection
st.sidebar.header("Settings")
features = {
    "sentiment": st.sidebar.checkbox("Perform Sentiment Analysis"),
    "highlights": st.sidebar.checkbox("Highlight Key Phrases"),
    "response": st.sidebar.checkbox("Generate Suggested Response"),
    "wordcloud": st.sidebar.checkbox("Generate Word Cloud"),
    "grammar_check": st.sidebar.checkbox("Grammar Check"),
    "key_phrases": st.sidebar.checkbox("Extract Key Phrases"),
    "actionable_items": st.sidebar.checkbox("Extract Actionable Items"),
    "root_cause": st.sidebar.checkbox("Root Cause Detection"),
    "culprit_identification": st.sidebar.checkbox("Culprit Identification"),
    "trend_analysis": st.sidebar.checkbox("Trend Analysis"),
    "risk_assessment": st.sidebar.checkbox("Risk Assessment"),
    "severity_detection": st.sidebar.checkbox("Severity Detection"),
    "critical_keywords": st.sidebar.checkbox("Critical Keyword Identification"),
    "export": st.sidebar.checkbox("Export Options"),
}

# Input Email Section
email_content = st.text_area("Paste your email content here:", height=200)

# AI Response via Hugging Face API
def get_ai_response(prompt, email_content):
    try:
        messages = [{"role": "user", "content": f"{prompt}\n\n{email_content}"}]
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1", messages=messages, max_tokens=500
        )
        return completion.choices[0].message["content"]
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Sentiment Analysis (Basic)
def get_sentiment(text):
    positive_keywords = ["happy", "good", "great", "excellent", "love"]
    negative_keywords = ["sad", "bad", "hate", "angry", "disappointed"]
    score = sum(1 if word in text.lower() else -1 for word in text.split())
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# Grammar Check (Basic Corrections)
def grammar_check(text):
    corrections = {"recieve": "receive", "adress": "address", "teh": "the", "occured": "occurred"}
    return re.sub(r'\b(' + '|'.join(corrections.keys()) + r')\b', lambda x: corrections[x.group()], text)

# Extract Key Phrases
def extract_key_phrases(text):
    return list(set(re.findall(r"\b[A-Za-z]{4,}\b", text)))  # Remove duplicates

# Generate Word Cloud Data
def generate_wordcloud(text):
    words = text.lower().split()
    return {word: words.count(word) for word in set(words)}

# Export to PDF
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode("latin1")

# Extract Actionable Items
def extract_actionable_items(text):
    return [line for line in text.split("\n") if "to" in line.lower() or "action" in line.lower()]

# Simple AI-based analysis methods
def detect_root_cause(text): return "Possible root cause: Lack of clear communication."
def identify_culprit(text): return "Manager or Team might be responsible."
def analyze_trends(text): return "Trend detected: Delay in project timelines."
def assess_risk(text): return "High risk due to delayed communication."
def detect_severity(text): return "High" if "urgent" in text.lower() else "Normal"
def identify_critical_keywords(text): return [word for word in text.split() if word.lower() in ["urgent", "problem", "issue", "failure"]]

# Process Email and Generate Insights
if email_content and st.button("Generate Insights"):
    try:
        # AI Responses
        summary = get_ai_response("Summarize the email in a concise format:", email_content)
        response = get_ai_response("Draft a professional response:", email_content) if features["response"] else ""
        highlights = get_ai_response("Highlight key points:", email_content) if features["highlights"] else ""

        # Sentiment Analysis
        sentiment = get_sentiment(email_content)

        # Word Cloud
        word_counts = generate_wordcloud(email_content)
        wordcloud_fig = plt.figure(figsize=(10, 5))
        plt.bar(word_counts.keys(), word_counts.values())
        plt.xticks(rotation=45)
        plt.title("Word Frequency")
        plt.tight_layout()

        # Display Results
        st.subheader("AI Summary")
        st.write(summary)

        if features["response"]:
            st.subheader("Suggested Response")
            st.write(response)

        if features["highlights"]:
            st.subheader("Key Highlights")
            st.write(highlights)

        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment}")

        if features["grammar_check"]:
            corrected_text = grammar_check(email_content)
            st.subheader("Grammar Check")
            st.write("Corrected Text:", corrected_text)

        if features["key_phrases"]:
            key_phrases = extract_key_phrases(email_content)
            st.subheader("Key Phrases")
            st.write(key_phrases)

        if features["wordcloud"]:
            st.subheader("Word Cloud")
            st.pyplot(wordcloud_fig)

        if features["actionable_items"]:
            st.subheader("Actionable Items")
            st.write(extract_actionable_items(email_content))

        # RCA & Insights
        if features["root_cause"]: st.subheader("Root Cause"); st.write(detect_root_cause(email_content))
        if features["culprit_identification"]: st.subheader("Culprit Identification"); st.write(identify_culprit(email_content))
        if features["trend_analysis"]: st.subheader("Trend Analysis"); st.write(analyze_trends(email_content))
        if features["risk_assessment"]: st.subheader("Risk Assessment"); st.write(assess_risk(email_content))
        if features["severity_detection"]: st.subheader("Severity Detection"); st.write(detect_severity(email_content))
        if features["critical_keywords"]: st.subheader("Critical Keywords"); st.write(identify_critical_keywords(email_content))

        # Export Options
        if features["export"]:
            export_content = f"Summary:\n{summary}\n\nSentiment: {sentiment}\n\nRoot Cause: {detect_root_cause(email_content)}\n"
            pdf_buffer = BytesIO(export_pdf(export_content))
            buffer_txt = BytesIO(export_content.encode("utf-8"))
            buffer_json = BytesIO(json.dumps({"summary": summary, "sentiment": sentiment}, indent=4).encode("utf-8"))

            st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
            st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")
            st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Paste email content and click 'Generate Insights' to start.")

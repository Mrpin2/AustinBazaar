import streamlit as st
from typing import Optional, List
import tempfile
import os
import json
import base64
from datetime import datetime

try:
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError:
    st.error("PyMuPDF (fitz) or Pillow not installed.")
    fitz = None
    Image = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from google import genai
except ImportError:
    genai = None

# --- Helper Functions ---
def convert_pdf_to_images(file_path):
    if fitz is None or Image is None:
        return []

    images = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            buffer = tempfile.SpooledTemporaryFile()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            b64_img = base64.b64encode(buffer.read()).decode("utf-8")
            images.append(f"data:image/png;base64,{b64_img}")
        doc.close()
    except Exception as e:
        st.error(f"Failed to convert PDF to images: {e}")
    return images

# --- Prompt Builder ---
def build_prompt(include_extra):
    if include_extra:
        return (
            "Extract all relevant invoice details you can find from the images. This includes, but is not limited to: "
            "File Name, Invoice Number, Invoice Date, Seller Name, Seller Address, Buyer Name, Buyer Address, "
            "Ship To Name, Ship To Address, Payment Terms, Shipping Method, Contact Info, PO Number, "
            "Line Items (with description, SKU if available, quantity, unit price, total, discount), Subtotal, Tax, Total Amount. "
            "Return structured data as JSON. If any detail is missing, use null."
        )
    else:
        return (
            "Extract the following details from each invoice in the images: "
            "File Name, Seller Name, Seller Address, Buyer Name, Buyer Address, Ship To Name, Ship To Address, "
            "Line Items (with description, SKU if available, quantity, amount). "
            "Return data as JSON. If anything is missing, use null."
        )

# --- Main App ---
st.set_page_config(page_title="US Invoice Extractor", layout="wide")
st.title("📄 US Equipment Invoice Extractor")

st.markdown("""
This app extracts structured data from US equipment-related invoices.

**You can choose to extract basic fields or let AI extract all possible fields it can detect.**
""")

st.sidebar.header("Configuration")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "Rajeev")
admin_input = st.sidebar.text_input("Admin Password (Optional):", type="password")

use_secrets = False
if admin_input:
    if admin_input == ADMIN_PASSWORD:
        st.sidebar.success("Admin mode activated. Using secret API keys.")
        use_secrets = True
    else:
        st.sidebar.error("Incorrect admin password.")

model_choice = st.sidebar.radio("Choose AI Model:", ("Google Gemini", "OpenAI GPT"))
include_extra_fields = st.sidebar.checkbox("Extract All Relevant Fields (AI-decided)", value=False)

if use_secrets:
    if model_choice == "Google Gemini":
        api_key = st.secrets.get("GEMINI_API_KEY")
        model_id = st.secrets.get("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
    elif model_choice == "OpenAI GPT":
        api_key = st.secrets.get("OPENAI_API_KEY")
        model_id = st.secrets.get("OPENAI_MODEL_ID", "gpt-4o")
else:
    if model_choice == "Google Gemini":
        api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
        model_id = st.sidebar.text_input("Gemini Model ID:", "gemini-1.5-flash-latest")
    elif model_choice == "OpenAI GPT":
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        model_id = st.sidebar.text_input("OpenAI Model ID:", "gpt-4o")

uploaded_files = st.file_uploader("Upload PDF invoices", type="pdf", accept_multiple_files=True)

if st.button("Extract Invoice Details"):
    if not api_key:
        st.error("API key not found. Please configure in secrets or input manually.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        prompt = build_prompt(include_extra_fields)
        for file in uploaded_files:
            st.subheader(f"📄 {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            try:
                if model_choice == "OpenAI GPT" and OpenAI:
                    client = OpenAI(api_key=api_key)
                    images = convert_pdf_to_images(tmp_path)
                    if not images:
                        st.error("Failed to render PDF pages.")
                        continue

                    message_content = [{"type": "text", "text": prompt}] + [
                        {"type": "image_url", "image_url": {"url": img, "detail": "high"}} for img in images
                    ]

                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": message_content}
                        ],
                        response_format="json"
                    )

                    raw_json = response.choices[0].message.content
                    st.json(json.loads(raw_json))

                elif model_choice == "Google Gemini" and genai:
                    client = genai.Client(api_key=api_key)
                    file_resource = client.files.upload(file=tmp_path, config={'display_name': file.name})
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[prompt, file_resource],
                        config={"response_mime_type": "application/json"}
                    )
                    st.json(response.candidates[0].content.parts[0].text)

                else:
                    st.error("Model client could not be initialized.")

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

            finally:
                os.remove(tmp_path)

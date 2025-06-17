import streamlit as st
from typing import Optional, List
import tempfile
import os
import json
import base64
from datetime import datetime
import re

try:
    import fitz  # PyMuPDF
    from PIL import Image
except ImportError:
    st.error("PyMuPDF (fitz) or Pillow not installed.")
    fitz = None
    Image = None

try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
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

# --- Prompts ---
INVOICE_EXTRACTION_PROMPT = (
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

**Extracted Fields:**
- File Name
- Seller Name and Address
- Buyer Name and Address
- Ship To Name and Address
- All line items with SKU, Quantity, Amount
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

if use_secrets:
    if model_choice == "Google Gemini":
        api_key = st.secrets.get("GEMINI_API_KEY")
        model_id = st.secrets.get("GEMINI_MODEL_ID", "gemini-1.5-flash-latest")
    elif model_choice == "OpenAI GPT":
        api_key = st.secrets.get("OPENAI_API_KEY")
        model_id = s

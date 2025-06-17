import streamlit as st
from typing import Optional, List
import tempfile
import os
import json
import base64
import pandas as pd
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

api_key = ""
model_id = ""

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
        for file in uploaded_files:
            st.subheader(f"📄 {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            try:
                extracted_data = None

                if model_choice == "OpenAI GPT" and openai:
                    openai.api_key = api_key
                    images = convert_pdf_to_images(tmp_path)
                    if not images:
                        st.error("Failed to render PDF pages.")
                        continue

                    message_content = [{"type": "text", "text": INVOICE_EXTRACTION_PROMPT}] + [
                        {"type": "image_url", "image_url": {"url": img, "detail": "high"}} for img in images
                    ]

                    response = openai.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": INVOICE_EXTRACTION_PROMPT},
                            {"role": "user", "content": message_content}
                        ]
                    )

                    raw_content = response.choices[0].message.content
                    json_candidate = re.search(r"\{.*\}", raw_content, re.DOTALL)
                    if json_candidate:
                        try:
                            extracted_data = json.loads(json_candidate.group())
                            st.json(extracted_data)
                        except json.JSONDecodeError:
                            st.error("OpenAI returned invalid JSON.")
                            st.text(raw_content)
                    else:
                        st.error("No JSON object found in response.")
                        st.text(raw_content)

                elif model_choice == "Google Gemini" and genai:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(model_id)
                    file_resource = genai.upload_file(path=tmp_path, display_name=file.name)
                    response = model.generate_content(
                        [INVOICE_EXTRACTION_PROMPT, file_resource],
                        generation_config={"response_mime_type": "application/json"}
                    )
                    try:
                        extracted_data = json.loads(response.text)
                        st.json(extracted_data)
                    except Exception:
                        st.error("Could not parse Gemini response.")
                        st.text(response.text)

                else:
                    st.error("Model client could not be initialized. Check API key and installation.")

                # --- Excel Export ---
                if extracted_data:
                    line_items = extracted_data.get("Line Items") or extracted_data.get("line_items")
                    if isinstance(line_items, list):
                        df = pd.DataFrame(line_items)
                        st.dataframe(df)

                        excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
                        df.to_excel(excel_buffer.name, index=False)
                        with open(excel_buffer.name, "rb") as f:
                            st.download_button(
                                label="📥 Download Line Items as Excel",
                                data=f.read(),
                                file_name=f"{file.name}_line_items.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

            finally:
                os.remove(tmp_path)

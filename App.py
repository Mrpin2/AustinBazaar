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
    st.error("PyMuPDF (fitz) or Pillow not installed. Please install them using 'pip install PyMuPDF Pillow'")
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
            # Render page to a high-resolution pixmap
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Save image to a BytesIO object and then base64 encode
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
    "Return data as JSON. If anything is missing, use null. "
    "Ensure the JSON output is a list of invoices, where each invoice is an object, "
    "and line items are an array of objects within each invoice. Example: "
    """
    {
      "invoices": [
        {
          "fileName": "invoice1.pdf",
          "sellerName": "Seller Co.",
          "sellerAddress": "123 Seller St.",
          "buyerName": "Buyer Inc.",
          "buyerAddress": "456 Buyer Ave.",
          "shipToName": "Recipient Ltd.",
          "shipToAddress": "789 Ship St.",
          "lineItems": [
            {"description": "Product A", "SKU": "PA-001", "quantity": 2, "amount": 100.00},
            {"description": "Service B", "SKU": null, "quantity": 1, "amount": 50.00}
          ]
        }
      ]
    }
    """
)


# --- Main App ---
st.set_page_config(page_title="US Invoice Extractor", layout="wide")
st.title("📄 US Equipment Invoice Extractor")

st.markdown("""
This app extracts structured data from US equipment-related invoices using AI.

**Extracted Fields:**
- File Name
- Seller Name and Address
- Buyer Name and Address
- Ship To Name and Address
- All line items with SKU, Quantity, Amount
""")

st.sidebar.header("Configuration")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "Rajeev") # Default value for local testing
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
        st.error("API key not found. Please configure in Streamlit secrets or input manually in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file to extract details.")
    else:
        all_extracted_line_items = [] # To accumulate all line items for a single Excel

        for file in uploaded_files:
            st.subheader(f"📄 Processing: {file.name}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name

            try:
                extracted_data = None

                with st.spinner(f"Analyzing {file.name}..."):
                    if model_choice == "OpenAI GPT" and openai:
                        if not api_key:
                            st.error("OpenAI API Key is missing.")
                            continue
                        openai.api_key = api_key
                        images = convert_pdf_to_images(tmp_path)
                        if not images:
                            st.error(f"Failed to render PDF pages for {file.name}. Please ensure PyMuPDF and Pillow are installed.")
                            continue

                        message_content = [{"type": "text", "text": INVOICE_EXTRACTION_PROMPT}] + [
                            {"type": "image_url", "image_url": {"url": img, "detail": "high"}} for img in images
                        ]

                        response = openai.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": INVOICE_EXTRACTION_PROMPT}, # System role for instruction
                                {"role": "user", "content": message_content}
                            ],
                            response_format={"type": "json_object"} # Instruct to return JSON
                        )

                        raw_content = response.choices[0].message.content
                        try:
                            extracted_data = json.loads(raw_content)
                        except json.JSONDecodeError as e:
                            st.error(f"OpenAI returned invalid JSON for {file.name}: {e}\nRaw content: {raw_content[:500]}...") # Show part of raw content for debug
                            st.info("Tip: If the JSON is malformed, try adjusting the prompt or model parameters.")

                    elif model_choice == "Google Gemini" and genai:
                        if not api_key:
                            st.error("Google Gemini API Key is missing.")
                            continue
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_id)

                        # For Gemini, directly upload the PDF
                        file_resource = genai.upload_file(path=tmp_path, display_name=file.name)
                        response = model.generate_content(
                            [INVOICE_EXTRACTION_PROMPT, file_resource],
                            generation_config={"response_mime_type": "application/json"}
                        )
                        try:
                            extracted_data = json.loads(response.text)
                        except Exception as e:
                            st.error(f"Could not parse Gemini response for {file.name}: {e}\nRaw content: {response.text[:500]}...")
                            st.info("Tip: If the JSON is malformed, try adjusting the prompt or model parameters.")
                        finally:
                            genai.delete_file(file_resource.name) # Clean up uploaded file

                # --- Display & Excel Export ---
                if extracted_data:
                    # Debugging line: Display the raw JSON response
                    st.subheader(f"Raw AI Output for {file.name}:")
                    st.json(extracted_data)
                    st.markdown("---")

                    invoices = extracted_data.get("invoices")
                    if invoices and isinstance(invoices, list):
                        for invoice in invoices:
                            st.markdown(f"**File Name:** {invoice.get('fileName', 'N/A')}")
                            st.markdown(f"**Seller:** {invoice.get('sellerName', 'N/A')}")
                            st.markdown(f"**Seller Address:** {invoice.get('sellerAddress', 'N/A')}")
                            st.markdown(f"**Buyer:** {invoice.get('buyerName', 'N/A')}")
                            st.markdown(f"**Buyer Address:** {invoice.get('buyerAddress', 'N/A')}")
                            st.markdown(f"**Ship To:** {invoice.get('shipToName', 'N/A')}")
                            st.markdown(f"**Ship To Address:** {invoice.get('shipToAddress', 'N/A')}")

                            line_items = invoice.get("lineItems", [])
                            if isinstance(line_items, list) and line_items:
                                st.subheader("Line Items:")
                                df = pd.DataFrame(line_items)
                                st.dataframe(df)
                                all_extracted_line_items.extend(line_items) # Collect for combined Excel

                            else:
                                st.info(f"No line items found for {file.name} in the extracted data.")
                            st.markdown("---")
                    else:
                        st.warning(f"No 'invoices' array found or it's empty in the AI's JSON output for {file.name}.")
                else:
                    st.error(f"No data extracted for {file.name}. Check API key, model ID, and prompt.")

            except Exception as e:
                st.error(f"An unexpected error occurred while processing {file.name}: {e}")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path) # Clean up the temporary PDF file

        # Download all extracted line items to a single Excel file
        if all_extracted_line_items:
            combined_df = pd.DataFrame(all_extracted_line_items)
            excel_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
            combined_df.to_excel(excel_buffer.name, index=False)
            with open(excel_buffer.name, "rb") as f:
                st.download_button(
                    label="⬇️ Download All Line Items as Excel",
                    data=f.read(),
                    file_name=f"all_invoices_line_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            os.remove(excel_buffer.name) # Clean up the temporary Excel file
        elif uploaded_files and all_extracted_line_items == []: # Only show if files were uploaded but no items found
            st.info("No line items were extracted from any of the uploaded invoices.")

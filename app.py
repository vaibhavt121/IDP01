import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import requests  # For direct API calls
import tempfile
import json
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Import EasyOCR after suppressing warnings
import easyocr

# Set page configuration
st.set_page_config(
    page_title="IndiAI IDP - Intelligent Document Processing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Simple OpenAI client that doesn't rely on the OpenAI library
class SimpleOpenAIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"

    def analyze_document(self, text, doc_type):
        """Use OpenAI API to analyze document text"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        prompt = f"""
        Analyze the following extracted text from a {doc_type} document.
        Extract key information like dates, names, amounts, addresses, and other relevant entities.
        Structure the output as JSON with appropriate fields for this document type.

        Extracted Text:
        {text}
        """

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system",
                 "content": "You are an expert document analyst that extracts structured information from documents."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            return json.loads(result["choices"][0]["message"]["content"])
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")


# Initialize session state variables
if 'extracted_text' not in st.session_state:
    st.session_state['extracted_text'] = ""
if 'analyzed_data' not in st.session_state:
    st.session_state['analyzed_data'] = {}
if 'reader' not in st.session_state:
    st.session_state['reader'] = None
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = 0.5
if 'openai_client' not in st.session_state:
    st.session_state['openai_client'] = None

# Sidebar for configuration
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=IndiAI+Logo", width=150)
    st.title("IndiAI IDP")

    # OpenAI API Key
    openai_api_key = st.text_input("API Key", type="password")
    if openai_api_key:
        # Use our simple client instead of the OpenAI SDK
        st.session_state['openai_client'] = SimpleOpenAIClient(api_key=openai_api_key)

    # Language selection for OCR
    language_options = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Marathi": "mr",
        "Tamil": "ta",
        "Telugu": "te",
        "Urdu": "ur"
    }

    selected_languages = st.multiselect(
        "Select Document Languages",
        options=list(language_options.keys()),
        default=["English"]
    )

    # Get language codes for EasyOCR
    language_codes = [language_options[lang] for lang in selected_languages]

    # OCR confidence threshold
    st.session_state['confidence_threshold'] = st.slider(
        "OCR Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )

    # Document type selection
    document_type = st.selectbox(
        "Document Type",
        options=["Invoice", "ID Card", "Resume", "Form", "Receipt", "Other"]
    )

    # Initialize the OCR reader
    if st.button("Initialize OCR Engine") or st.session_state['reader'] is not None:
        with st.spinner("Loading OCR engine..."):
            if st.session_state['reader'] is None and language_codes:
                st.session_state['reader'] = easyocr.Reader(language_codes)
            st.success("OCR engine loaded successfully!")

# Main app
st.title("IndiAI IDP - Intelligent Document Processing")

# Upload document
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "png", "jpg", "jpeg"])

# Process the document
if uploaded_file is not None:
    # Display the uploaded document
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Document")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Display the image
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, width=400)
            # Convert PIL Image to numpy array for OCR
            img_array = np.array(image)
        elif uploaded_file.type == 'application/pdf':
            # For PDFs, display a placeholder initially
            st.write("PDF document uploaded. Click 'Extract Text' to process.")
            img_array = None  # Will be set during extraction

    # Perform OCR extraction if the reader is initialized
    if st.session_state['reader'] is not None and st.button("Extract Text"):
        with st.spinner("Extracting text from document..."):
            if uploaded_file.type.startswith('image'):
                # Process with EasyOCR
                results = st.session_state['reader'].readtext(img_array)

                # Filter results by confidence
                filtered_results = [r for r in results if r[2] >= st.session_state['confidence_threshold']]

                # Visualize OCR results
                viz_img = Image.fromarray(img_array.copy())
                draw = ImageDraw.Draw(viz_img)

                for (bbox, text, prob) in filtered_results:
                    # Draw bounding box
                    draw.polygon([tuple(p) for p in bbox], outline="red")
                    # Draw text
                    draw.text((bbox[0][0], bbox[0][1] - 10), f"{text} ({prob:.2f})", fill="red")

                with col1:
                    st.image(viz_img, width=400, caption="OCR Results")

                # Extract and format the text
                extracted_text = "\n".join([r[1] for r in filtered_results])
                st.session_state['extracted_text'] = extracted_text

            elif uploaded_file.type == 'application/pdf':
                # For PDFs, we need to convert to images first
                try:
                    from pdf2image import convert_from_bytes

                    # Convert PDF to images
                    images = convert_from_bytes(uploaded_file.getvalue())

                    if len(images) > 0:
                        # Display first page
                        st.image(images[0], width=400, caption="First page of PDF")

                        # Process first page for OCR
                        first_page = np.array(images[0])
                        img_array = first_page  # Use first page for OCR

                        # Process with EasyOCR
                        results = st.session_state['reader'].readtext(img_array)

                        # Filter results by confidence
                        filtered_results = [r for r in results if r[2] >= st.session_state['confidence_threshold']]

                        # Visualize OCR results
                        viz_img = Image.fromarray(img_array.copy())
                        draw = ImageDraw.Draw(viz_img)

                        for (bbox, text, prob) in filtered_results:
                            # Draw bounding box
                            draw.polygon([tuple(p) for p in bbox], outline="red")
                            # Draw text
                            draw.text((bbox[0][0], bbox[0][1] - 10), f"{text} ({prob:.2f})", fill="red")

                        with col1:
                            st.image(viz_img, width=400, caption="OCR Results")

                        # Extract and format the text
                        extracted_text = "\n".join([r[1] for r in filtered_results])
                        st.session_state['extracted_text'] = extracted_text

                        # Show total pages info if more than one page
                        if len(images) > 1:
                            st.info(f"PDF has {len(images)} pages. Currently showing first page only.")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state['extracted_text'] = "Error processing PDF."

    # Display extracted text
    with col2:
        st.subheader("Extracted Text")
        st.text_area("Extracted Text Content", st.session_state['extracted_text'], height=400,
                     label_visibility="collapsed")

    # Analyze with OpenAI GPT if API key is provided
    if st.session_state['openai_client'] and st.session_state['extracted_text'] and st.button("Analyze with AI"):
        with st.spinner("Analyzing document with AI..."):
            try:
                # Use our simple client to call OpenAI
                analyzed_data = st.session_state['openai_client'].analyze_document(
                    st.session_state['extracted_text'],
                    document_type
                )
                st.session_state['analyzed_data'] = analyzed_data

            except Exception as e:
                st.error(f"Error during AI analysis: {str(e)}")

    # Display analyzed data
    if st.session_state['analyzed_data']:
        st.subheader("AI Analysis Results")

        # Create a cleaner display for the analysis results
        col1, col2 = st.columns(2)

        with col1:
            st.json(st.session_state['analyzed_data'])

        with col2:
            # Display in a more user-friendly format
            st.subheader("Extracted Entities")
            for key, value in st.session_state['analyzed_data'].items():
                if isinstance(value, dict):
                    st.write(f"**{key.replace('_', ' ').title()}**")
                    for sub_key, sub_value in value.items():
                        st.write(f"- {sub_key.replace('_', ' ').title()}: {sub_value}")
                else:
                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")

        # Add export options
        st.download_button(
            label="Export as JSON",
            data=json.dumps(st.session_state['analyzed_data'], indent=2),
            file_name="analysis_results.json",
            mime="application/json"
        )

        # Display as CSV if the data structure allows
        try:
            flat_data = pd.json_normalize(st.session_state['analyzed_data'])
            st.download_button(
                label="Export as CSV",
                data=flat_data.to_csv(index=False),
                file_name="analysis_results.csv",
                mime="text/csv"
            )
        except:
            st.write("CSV export not available for this document structure")

# About section
st.sidebar.markdown("---")
st.sidebar.subheader("About IndiAI IDP")
st.sidebar.info("""
IndiAI IDP is an intelligent document processing application that implements advanced document analysis. 
It supports multiple Indian languages and various document types.
""")

# Clean up temporary files
try:
    if 'temp_file_path' in locals():
        os.unlink(temp_file_path)
except:
    pass
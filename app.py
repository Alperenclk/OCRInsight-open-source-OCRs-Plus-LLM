import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import tempfile
import os
import time
import io
import json
import torch
import cv2

# Import OCR engines
import ocr_engines

# Try importing LLM processor if LLM features are to be used
llm_available = False
try:
    import llm_processor

    llm_available = True
except ImportError:
    pass  # LLM features will be disabled

# Create results folder if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

# Streamlit application
st.title("OCRInsight")

# Sidebar
st.sidebar.header("Settings")


# Function to save text to file
def save_text_to_file(attributes_of_output, all_ocr_text, filename):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n" + "-" * 75 + "\n")
        f.write("Attributes of Output:\n")
        f.write(attributes_of_output)
        f.write("\nOCR Result:\n")
        f.write(all_ocr_text)
        f.write("\n" + "-" * 75 + "\n")
    st.success(f"{filename} saved successfully!")


# Device selection
device = st.sidebar.radio("Select Device", ["CPU", "GPU (CUDA)"])
save_output = st.sidebar.checkbox("Save Outputs")

# Language selection
language = st.sidebar.selectbox(
    "Select Language", ["Türkçe", "English", "Français", "Deutsch", "Español"]
)

# Map selected language to language codes
language_codes = {
    "Türkçe": "tr",
    "English": "en",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es",
}

# OCR model selection
ocr_models = st.sidebar.multiselect(
    "Select OCR Models",
    ["EasyOCR", "DocTR", "Tesseract", "PaddleOCR"],
    ["EasyOCR"],  # default selection
)

# LLM model selection
llm_model = st.sidebar.selectbox(
    "Select LLM Model", ["Only OCR Mode", "llama3.1", "llama3", "gemma2"]
)

# Conditional UI elements based on LLM model selection
if llm_model != "Only OCR Mode" and llm_available:
    user_command = st.sidebar.text_input("Enter command:", "")

    task_type = st.sidebar.radio("Select task type:", ["Summarize", "Generate"])
elif llm_model != "Only OCR Mode" and not llm_available:
    st.sidebar.warning(
        "LLM features are not available. Please install 'ollama' to enable LLM processing."
    )
    llm_model = "Only OCR Mode"

# Check GPU availability
if device == "GPU (CUDA)" and not torch.cuda.is_available():
    st.sidebar.warning("GPU (CUDA) not available. Switching to CPU.")
    device = "CPU"

# Initialize OCR models
ocr_readers = ocr_engines.initialize_ocr_models(
    ocr_models, language_codes[language], device
)

# File upload
uploaded_file = st.file_uploader(
    "Upload File (PDF, Image)", type=["pdf", "png", "jpg", "jpeg"]
)

# Create results folder if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

if uploaded_file is not None:
    start_time = time.time()

    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        total_pages = len(pdf_document)
        pdf_document.close()
    else:
        images = [Image.open(uploaded_file)]
        total_pages = 1

    all_ocr_texts = {
        model_name: "" for model_name in ocr_models
    }  # To store OCR text for each model

    for page_num, image in enumerate(images, start=1):
        st.image(image, caption=f"Page {page_num}/{total_pages}", use_column_width=True)

        # Perform OCR with each selected model
        for model_name in ocr_models:
            text = ocr_engines.perform_ocr(
                model_name, ocr_readers, image, language_codes[language]
            )
            all_ocr_texts[
                model_name
            ] += f"--- Page {page_num} ({model_name}) ---\n{text}\n\n"

            st.subheader(f"OCR Result ({model_name}) - Page {page_num}/{total_pages}:")
            st.text(text)

    end_time = time.time()
    process_time = end_time - start_time

    st.info(f"Processing time: {process_time:.2f} seconds")

    # Save OCR outputs if selected
    if save_output:
        attributes_of_output = {
            "Model Names": ocr_models,
            "Language": language,
            "Device": device,
            "Process Time": process_time,
        }
        for model_name, ocr_text in all_ocr_texts.items():
            filename = f"results//ocr_output_{model_name}.txt"
            save_text_to_file(
                json.dumps(attributes_of_output, ensure_ascii=False), ocr_text, filename
            )

    # LLM processing
    if (
        llm_model != "Only OCR Mode"
        and llm_available
        and st.sidebar.button("Start LLM Processing")
    ):
        st.subheader("LLM Processing Result:")

        # Combine all OCR texts
        combined_ocr_text = "\n".join(all_ocr_texts.values())

        # Prepare the prompt based on the task type
        if task_type == "Summarize":
            prompt = f"Please summarize the following text. Command: {user_command}\n\nText: {combined_ocr_text}"
        else:  # "Generate"
            prompt = f"Please generate new text based on the following text. Command: {user_command}\n\nText: {combined_ocr_text}"

        llm_output = llm_processor.process_with_llm(llm_model, prompt)

        # Display the result
        st.write(f"Processing completed using '{llm_model}' model.")
        st.text_area("LLM Output:", value=llm_output, height=300)

        # Save LLM output if selected
        if save_output:
            filename = "llm_output.txt"
            save_text_to_file(llm_output, "", filename)

elif llm_model != "Only OCR Mode" and not llm_available:
    st.warning(
        "LLM features are not available. Please install 'ollama' to enable LLM processing."
    )

st.sidebar.info(f"Selected device: {device}")

# OCR and LLM Application

This Streamlit application combines Optical Character Recognition (OCR) and Language Model (LLM) capabilities to process and analyze documents and images.

## Features

- Supports multiple OCR models: Tesseract, EasyOCR, and DocTR
- Handles PDF and image file inputs
- Multi-language support
- GPU acceleration option
- Integration with LLM models (planned feature)

## Requirements

- Python 3.7+
- Streamlit
- PyTesseract
- Pillow
- pdf2image
- EasyOCR
- OpenCV
- NumPy
- PyTorch
- python-doctr

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Usage

1. Select device (CPU/GPU) and language
2. Choose OCR and LLM models
3. Upload a PDF or image file
4. View OCR results and processing time
5. (Future feature) Summarize or edit text using LLM

## Note

LLM integration with Ollama is planned but not yet implemented.
import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import easyocr
import cv2
import numpy as np
import requests
import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import tempfile
import os
import time

# Streamlit uygulamasını oluştur
st.title("OCR ve LLM Uygulaması")

# Sol kenar çubuğu
st.sidebar.header("Ayarlar")

# Cihaz seçimi
device = st.sidebar.radio(
    "Cihaz Seçin",
    ["CPU", "GPU (CUDA)"]
)

# Language selection
language = st.sidebar.selectbox(
    "Dil Seçin",
    ["Türkçe", "English", "Français", "Deutsch", "Español"]
)

# Map selected language to language codes
language_codes = {
    "Türkçe": "tr",
    "English": "en",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es"
}

# OCR model seçimi
ocr_model = st.sidebar.selectbox(
    "OCR Modeli Seçin",
    ["Tesseract", "EasyOCR", "DocTR"]
)

# LLM model seçimi
llm_model = st.sidebar.selectbox(
    "LLM Modeli Seçin",
    ["llama3.1", "llama3", "gemma2"]
)

# GPU kullanılabilirliğini kontrol et
if device == "GPU (CUDA)" and not torch.cuda.is_available():
    st.sidebar.warning("GPU (CUDA) kullanılamıyor. CPU'ya geçiliyor.")
    device = "CPU"

# OCR modellerini yükle
if ocr_model == "Tesseract":
    # Tesseract doesn't require explicit initialization
    pass
elif ocr_model == "EasyOCR":
    reader = easyocr.Reader([language_codes[language]], gpu=(device == "GPU (CUDA)"))
elif ocr_model == "DocTR":
    doctr_model = ocr_predictor(pretrained=True)

# Dosya yükleme
uploaded_file = st.file_uploader("Dosya Yükleyin (PDF, Resim)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    start_time = time.time()
    
    if uploaded_file.type == "application/pdf":
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        total_pages = len(images)
    else:
        images = [Image.open(uploaded_file)]
        total_pages = 1
    
    for page_num, image in enumerate(images, start=1):
        st.image(image, caption=f"Sayfa {page_num}/{total_pages}", use_column_width=True)
        
        # OCR işlemi
        if ocr_model == "Tesseract":
            text = pytesseract.image_to_string(image, lang=language_codes[language])
        elif ocr_model == "EasyOCR":
            result = reader.readtext(np.array(image))
            text = "\n".join([res[1] for res in result])
        elif ocr_model == "DocTR":  # DocTR
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                image.save(tmp_file, format=image.format)
            
            file_path = tmp_file.name
            doc = DocumentFile.from_images(file_path)
            result = doctr_model(doc)
            text = "\n\n".join([
                "\n".join([
                    " ".join([word.value for word in line.words])
                    for line in block.lines
                ])
                for block in result.pages[0].blocks
            ])
            
            os.unlink(file_path)
        
        st.subheader(f"OCR Sonucu (Sayfa {page_num}/{total_pages}):")
        st.text(text)
    
    end_time = time.time()
    process_time = end_time - start_time
    
    st.info(f"İşlem süresi: {process_time:.2f} saniye")
    
    # LLM işlemi
    if st.button("Metni Özetle veya Düzenle"):
        # Burada Ollama'ya istek gönderme işlemi yapılacak
        # Örnek olarak:
        # response = requests.post("http://localhost:11434/api/generate", json={
        #     "model": llm_model,
        #     "prompt": f"Özet: {text}"
        # })
        # summary = response.json()["response"]
        # st.subheader("Özet:")
        # st.write(summary)
        st.warning("LLM işlemi henüz uygulanmadı. Ollama entegrasyonu gerekiyor.")

st.sidebar.info(f"Seçilen cihaz: {device}")

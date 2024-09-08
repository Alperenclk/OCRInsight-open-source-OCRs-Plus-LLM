import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
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
import io
import json
import ollama

# Streamlit uygulamasını oluştur
st.title("OCR ve LLM Uygulaması")

# Sol kenar çubuğu
st.sidebar.header("Ayarlar")



# Function to save text to file
def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        f.write("-"*50)
    st.success(f"{filename} başarıyla kaydedildi!")

# Cihaz seçimi
device = st.sidebar.radio(
    "Cihaz Seçin",
    ["CPU", "GPU (CUDA)"]
)
save_output = st.sidebar.checkbox("Çıktıları Kaydet")

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
    [ "EasyOCR", "DocTR"]
)

# LLM model seçimi
llm_model = st.sidebar.selectbox(
    "LLM Modeli Seçin",
    ["Only OCR Mode", "llama3.1", "llama3", "gemma2"]
)


# Conditional UI elements based on LLM model selection
if llm_model != "Only OCR Mode":
    user_command = st.sidebar.text_input("Komut girin:", "")
    
    task_type = st.sidebar.radio(
        "İşlem türünü seçin:",
        ["Özetle", "Oluştur"]
    )

# GPU kullanılabilirliğini kontrol et
if device == "GPU (CUDA)" and not torch.cuda.is_available():
    st.sidebar.warning("GPU (CUDA) kullanılamıyor. CPU'ya geçiliyor.")
    device = "CPU"


if ocr_model == "EasyOCR":
    reader = easyocr.Reader([language_codes[language]], gpu=(device == "GPU (CUDA)"))
elif ocr_model == "DocTR":
    doctr_model = ocr_predictor(pretrained=True)

# Dosya yükleme
uploaded_file = st.file_uploader("Dosya Yükleyin (PDF, Resim)", type=["pdf", "png", "jpg", "jpeg"])

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
    
    all_ocr_text = ""  # To store all OCR text
    for page_num, image in enumerate(images, start=1):
        st.image(image, caption=f"Sayfa {page_num}/{total_pages}", use_column_width=True)
        
        if ocr_model == "EasyOCR":
            result = reader.readtext(np.array(image))
            text = "\n".join([res[1] for res in result])
        elif ocr_model == "DocTR":  # DocTR
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                image.save(tmp_file, format="PNG")
            
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
        
        all_ocr_text += f"--- Sayfa {page_num} ---\n{text}\n\n"
        
        st.subheader(f"OCR Sonucu (Sayfa {page_num}/{total_pages}):")
        st.text(text)
    
    end_time = time.time()
    process_time = end_time - start_time
    
    st.info(f"İşlem süresi: {process_time:.2f} saniye")
    
    # Save OCR output if selected

    if save_output:
        save_text_to_file(all_ocr_text, "ocr_output.txt")

    # LLM processing
    if llm_model != "Only OCR Mode" and st.sidebar.button("LLM İşlemini Başlat"):
        st.subheader("LLM İşlem Sonucu:")
        
        # Prepare the prompt based on the task type
        if task_type == "Özetle":
            prompt = f"Lütfen aşağıdaki metni özetle. Komut: {user_command}\n\nMetin: {all_ocr_text}"
        else:  # "Oluştur"
            prompt = f"Lütfen aşağıdaki metne dayanarak yeni bir metin oluştur. Komut: {user_command}\n\nMetin: {all_ocr_text}"
        

        response = ollama.chat(model=llm_model, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
            ])


        llm_output = result['message']['content']
        
        # Display the result
        st.write(f"'{llm_model}' modeli kullanılarak işlem tamamlandı.")
        st.text_area("LLM Çıktısı:", value=llm_output, height=300)
        
        # Save LLM output if selected
        if save_output:
            save_text_to_file(llm_output, "llm_output.txt")
            

st.sidebar.info(f"Seçilen cihaz: {device}")

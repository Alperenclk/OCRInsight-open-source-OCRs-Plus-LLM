# OCR and LLM Application
This Streamlit application allows users to perform OCR (Optical Character Recognition) using multiple open-source OCR engines and optionally process the OCR results using LLMs (Large Language Models). Users can compare the outputs of different OCR models and perform tasks such as summarization or text generation based on the OCR results.



![Test Image 4](https://github.com/Alperenclk/All_OCR-s_tools/sample_files/sample_screen.png 112)


## Features
### Multiple OCR Engines Supported:

* EasyOCR
* DocTR
* Tesseract OCR
* PaddleOCR

#### Optional LLM Processing:

Use models like llama3.1, llama3, gemma2 via Ollama.
Perform tasks such as summarization or text generation based on OCR results.

#### Compare OCR Outputs:

Select multiple OCR models to compare their outputs side by side.
#### Save Outputs:

Option to save OCR and LLM outputs to text files.

## Installation
### Prerequisites
- Python 3.7 or higher
- pip package manager

### Clone the Repository

```bash 
git clone https://github.com/yourusername/ocr-llm-app.git
cd ocr-llm-app
```

### Create a Virtual Environment (Recommended)
```bash 
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Required Python Packages
#### Install the required packages using pip:

```bash 
pip install -r requirements.txt
```
Note: The requirements.txt file includes basic dependencies. Depending on the OCR engines and LLM support you want to use, you may need to install additional dependencies as described below.

## Install OCR Engine Dependencies

### EasyOCR
```bash 
pip install easyocr
```

### DocTR
```bash 
pip install python-doctr[torch]
```
Note: For GPU support, ensure that PyTorch is installed with CUDA support.

### Tesseract OCR
Install Tesseract OCR Engine:

#### Windows:

Download the Tesseract installer from UB Mannheim: <https://github.com/UB-Mannheim/tesseract/wiki>.

**Run the installer and follow the instructions.
Note the installation path (e.g., C:\Program Files\Tesseract-OCR\tesseract.exe).
Update the pytesseract.pytesseract.tesseract_cmd variable in ocr_engines.py to point to the Tesseract executable.**

#### macOS:

```bash 
brew install tesseract
```
#### Ubuntu/Linux:

``` bash 
sudo apt-get update
sudo apt-get install tesseract-ocr
```

##### Install Python Wrapper:

```bash 
pip install pytesseract
```
##### Language Data Files:

Ensure that the language data files for the languages you intend to use are installed. For example, to install Turkish language data on Ubuntu:

```bash 
sudo apt-get install tesseract-ocr-tur
```

### PaddleOCR
#### Install PaddlePaddle:

#### CPU Version:

```bash 
pip install paddlepaddle
```
#### GPU Version:

Refer to the PaddlePaddle Installation Guide for GPU support.

### Install PaddleOCR:

```bash 
pip install paddleocr
```

## Install LLM Dependencies (Optional)
If you want to use the LLM features, install **Ollama**:

```bash
pip install ollama
```
Note: If you do not wish to use the LLM features, **you can skip this step**. The application will work in OCR-only mode.

## Usage
### Run the Application
```bash 
streamlit run app.py
```

## Application Interface
### Settings Sidebar:

**Select Device:** Choose between CPU and GPU (if available).

**Language Selection:** Choose the language for OCR processing.

**Select OCR Models:** Choose one or more OCR models to use.

**LLM Model Selection:** Choose an LLM model or select "Only OCR Mode" to disable LLM features.

**LLM Command and Task Type:** Enter commands and select tasks if LLM is enabled.

**Save Outputs:** Option to save OCR and LLM outputs to files.

### Main Area:

**File Upload:** Upload a PDF or image file for OCR processing.

**OCR Results:** View the OCR results from the selected models.

**LLM Processing:** Perform LLM processing on the combined OCR text (if enabled).
## Notes
**Language Support:**

Ensure that the necessary language data files or models are installed for each OCR engine you intend to use.
Some OCR engines may require specific language codes or configurations.

**GPU Support:**

For GPU acceleration, ensure that your hardware supports it and that the necessary libraries (e.g., CUDA) are installed.
Not all OCR engines support GPU acceleration.

**Performance:**

Processing multiple OCR engines simultaneously may consume significant resources.
Processing large files or images may take longer.
Modular Code Structure
The application is structured modularly to enhance maintainability and extensibility.

**app.py:** The main Streamlit application script.

**ocr_engines.py:** Contains functions to initialize and perform OCR using different engines.

**llm_processor.py:** Contains functions for LLM processing (optional).
Modifying the Code

#### **Adding a New OCR Engine:**

Create a new function in ocr_engines.py to initialize and perform OCR with the new engine.
Update initialize_ocr_models and perform_ocr functions accordingly.

**Modifying LLM Functionality:**

Update llm_processor.py with new LLM models or processing methods.

**Disabling LLM Features:**

If you don't want to use LLM features, you don't need to install ollama.
The application will automatically disable LLM features if ollama is not installed.

## Troubleshooting
**Import Errors:**

If you encounter import errors, ensure that all required packages are installed.
For optional features (like LLM), missing packages will disable those features without affecting the rest of the application.

**Tesseract Not Found:**

Ensure that the Tesseract executable path is correctly set in ocr_engines.py.
Verify that Tesseract is installed and the path is correct.

**Language Data Missing:**

Install the necessary language data files for the OCR engines.
Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or new features.

### License
This project is licensed under the **MIT** License.

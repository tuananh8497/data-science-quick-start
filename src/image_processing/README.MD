# 📷 Py-Ollama: Image Processing with AI Model Integration
## 📌 Overview
This script automates the processing of PNG images in a directory using Tesseract OCR to extract text, then generates prompts to interact with an Ollama AI model (e.g., gemma3:1b). It logs individual responses and compiles them into a markdown file for easy review.

## 📦 Requirements
Install dependencies via pip:
```sh
pip install pytesseract pillow ollama
```

Ensure the following Python packages are in your environment:

pytesseract (OCR)
Pillow (image processing)
ollama (AI model interaction)
python-dotenv (for loading environment variables)
## 📁 Setup
Create config.env with environment variables:

Ensure prompt templates exist in the prompts directory (e.g., template_image_to_text.txt).

## 🚀 Usage
1. Run via Make
2. What Happens
Processes all .png files in OLAMA_INPUT_DIR.
Extracts text using Tesseract OCR.
Generates prompts using the specified template.
Sends prompts to the Ollama model and logs responses.
Saves individual logs to OLAMA_LOG_DIR and compiles them into OLAMA_OUTPUT_FILE.
## 📁 Output Structure
Individual logs:
Saved as data/output/logs/log_<timestamp>.md with OCR text, prompts, and model responses.
Aggregated markdown:
Output at questions.md, sorted by question numbers.
## 📝 Notes
If OCR fails for an image, it skips processing (logs a warning).
All paths are resolved via environment variables for flexibility.
The script uses make for simplicity—customize Makefile if needed.
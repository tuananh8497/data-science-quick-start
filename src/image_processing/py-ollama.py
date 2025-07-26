from ollama import chat
import glob
from PIL import Image
import pytesseract
import time
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log_dir = Path('./data/output')
log_dir.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = 'gemma3:12b'

# Step 1: Get all PNG files
image_paths = glob.glob('./data/input/*.png')

# Step 2: Extract text using OCR
extracted_texts = []
for path in image_paths:
    image = Image.open(path)
    text = pytesseract.image_to_string(image)
    extracted_texts.append((path, text))

# Step 3: Ask model to respond in desired format
instruction = """
Format the input as:

Question <number>: <Extracted question>

Options:
1. ...
2. ...
3. ...
...

Answer: <correct option number>

Explanation: <brief explanation>

"""

# Step 4: Send prompt to model and dump response to log file
for path, text in extracted_texts:
    logging.info(f"\n🔹 Processing: {path}")
    if text.strip():
        full_prompt = f"{instruction}\n\n{text}"
        logging.debug("📜 Prompt:", full_prompt)
        response = chat(model=DEFAULT_MODEL, messages=[
            # {"role": "system", "content": "Always format output as: Question <number>, Options, Answer, Explanation."},
            {"role": "user", "content": f"{full_prompt}"},
        ])

        ts = int(time.time())
        log_path = log_dir / f'log_{ts}.md'
        
        logging.info(f"Model: {response.model}")
        logging.info(f"Inference Time (ms): {response.total_duration / 1_000_000:.2f}")
        logging.info(f"Prompt Tokens/s: {response.prompt_eval_count}/s")
        logging.info(f"Response Tokens/s: {response.eval_count}/s")

        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"Created at: {response.created_at}\n")
                f.write(f"Image path: `{path}`\n")
                f.write(f"# Response\n```text\n{response.message.content.strip()}\n```\n")
        except Exception as e:
            logging.error(f"Error writing to log file {log_path}: {e}")

    else:
        logging.warning("No text found in image.")

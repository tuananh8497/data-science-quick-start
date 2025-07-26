import logging
import time
import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

import pytesseract
from PIL import Image
from ollama import chat

# --- Config ---

@dataclass
class Config:
    model: str = "gemma3:1b"
    input_dir: Path = Path("./data/input")
    log_dir: Path = Path("./data/output/logs")
    output_file: Path = Path("./data/output/aggregated/questions.md")


# --- Logging ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# --- Utilities ---

def ocr_image(image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"OCR failed for {image_path}: {e}")
        return ""

def load_prompt_template(name: str) -> str:
    path = Path(f"./prompts/{name}.txt")
    if not path.exists():
        raise FileNotFoundError(f"Prompt template '{name}' not found.")
    return path.read_text(encoding='utf-8')
    

def build_prompt(ocr_text: str, prompt_template: str) -> str:
    instruction = load_prompt_template(prompt_template)
    return f"{instruction.strip()}\n\n{ocr_text.strip()}"


def call_model(prompt: str, model: str) -> str:
    response = chat(model=model, messages=[{"role": "user", "content": prompt}])
    logging.info(f"Inference Time (ms): {response.total_duration / 1_000_000:.2f}")
    logging.info(f"Prompt Tokens: {response.prompt_eval_count}")
    logging.info(f"Response Tokens: {response.eval_count}")
    if response.total_duration > 0:
        rps = response.eval_count / (response.total_duration / 1_000_000_000)  # ns to s
        logging.info(f"Response Speed: {rps:.2f} tokens/sec")    
    return response.message.content.strip()


def extract_question_number(text: str) -> int:
    match = re.search(r'Question\s*(\d+):', text)
    return int(match.group(1)) if match else 0


def write_response_log(path: Path, response: str, image_path: Path, created_at: str):
    try:
        path.write_text(
            f"Created at: {created_at}\nImage: `{image_path}`\n\n# Response\n```text\n{response}\n```",
            encoding="utf-8"
        )
    except Exception as e:
        logging.error(f"Failed to write log: {e}")


def generate_markdown(ollama_outputs: List[Tuple[Path, str]], output_path: Path):
    sorted_responses = sorted(ollama_outputs, key=lambda x: extract_question_number(x[1]))
    lines = []
    for img_path, response in sorted_responses:
        lines.append(response.strip())
        lines.append("\n---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"✅ Markdown written to {output_path}")


# --- Main ---

def main():
    cfg = Config(
        model="gemma3:12b",
    )
    logging.info(f"Using model: {cfg.model}")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(cfg.input_dir.glob("*.png"))
    ollama_outputs = []

    for img_path in image_paths:
        logging.info(f"📷 Processing: {img_path}")
        ocr_text = ocr_image(img_path)
        if not ocr_text.strip():
            logging.warning("⚠️ No text found.")
            continue

        prompt = build_prompt(ocr_text, prompt_template="template_image_to_text")
        response = call_model(prompt, cfg.model)

        # Write individual log
        ts = int(time.time())
        log_path = cfg.log_dir / f"log_{ts}.md"
        write_response_log(log_path, response, img_path, time.strftime('%Y-%m-%d %H:%M:%S'))

        ollama_outputs.append((img_path, response))

    generate_markdown(ollama_outputs, cfg.output_file)


if __name__ == "__main__":
    main()

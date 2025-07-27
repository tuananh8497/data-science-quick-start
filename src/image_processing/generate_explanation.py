# generate_explanation.py
import logging
from pathlib import Path
from ollama import chat
import re
from dataclasses import dataclass

# --- Config ---
@dataclass
class Config:
    input_file: Path = Path("./data/output/aggregated/questions.md")
    output_file: Path = Path("./data/output/aggregated/explanations.md")
    model: str = "gemma3:12b"
    instruction_file: Path = Path("./prompts/explain_spark_answer.txt")


# --- Main ---
def main():
    config = Config()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Read the input markdown file
    with open(config.input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Read the instruction file
    with open(config.instruction_file, "r", encoding="utf-8") as f:
        instruction = f.read().strip()

    # Split into question-answer pairs using "---" as separators
    sections = content.split("\n---\n")
    explanations = []

    for section in sections:
        # Extract question number (assuming format like "Question 1:")
        question_match = re.search(r"Question\s+(\d+):", section)
        if not question_match:
            logging.warning("⚠️ No question number found in section.")
            continue

        question_number = question_match.group(1)
        question_options = section.strip()
        print(question_options)

        # Generate explanation using the model
        prompt = f"{instruction}\n\nExplain the following answer: {question_options}"
        response = chat(model=config.model, messages=[{"role": "user", "content": prompt}])
        explanation = response.message.content.strip()

        # Format the output
        explanations.append(f"{question_options}\n\n{explanation}\n\n")


    # Write the output to a new markdown file
    with open(config.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(explanations))

    logging.info(f"✅ Explanations written to {config.output_file}")


if __name__ == "__main__":
    main()
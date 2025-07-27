import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def log_response_stats(response):
    """Log statistics from an Ollama model response."""
    logging.info(f"Inference Time (ms): {response.total_duration / 1_000_000:.2f}")
    logging.info(f"Prompt Tokens: {response.prompt_eval_count}")
    logging.info(f"Response Tokens: {response.eval_count}")
    if response.total_duration > 0:
        rps = response.eval_count / (response.total_duration / 1_000_000_000)  # ns to s
        logging.info(f"Response Speed: {rps:.2f} tokens/sec")
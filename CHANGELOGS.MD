## [2025-07-27] Refactor and Environment-based Configuration

### Changed
- Refactored `py-ollama.py` and `generate_explanation.py` to load configuration values (model, input/output/log directories, prompt template) from environment variables using `local_setup/config.env`.
- Updated `Config` dataclass initialization to use `os.getenv` for all configurable parameters, improving flexibility and consistency.
- Standardized use of `Path` objects for file and directory paths after loading from environment variables.
- Moved reusable response statistics logging to `src/utils/logging.py` and imported it where needed.
- Centralized logging configuration in the logging utility module.
- Added error handling for file operations and OCR failures.
- Updated Makefile and run instructions to support environment-

## [2025-07-26] Image Processing and Model Integration

### Added
- New script `py-ollama.py` for automated processing of PNG images in the `data/input` directory.
- Integration with Tesseract OCR to extract text content from images.
- Automated prompt generation for chat model interaction using extracted text.
- Logging of model responses, inference statistics, and metadata to timestamped markdown files in `data/output`.
- Automatic creation of the output directory if it does not exist.
- Enhanced error handling for file operations to improve robustness.

### Changed
- Improved logging throughout the script for better traceability and debugging
- Add pillow, pytesseract and ollama to requirements.txt
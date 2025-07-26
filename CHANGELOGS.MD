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
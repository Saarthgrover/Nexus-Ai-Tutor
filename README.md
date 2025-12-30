# Nexus-Ai-Tutor
Ai tutor that helps explain topics, organize notes, and much more

## Image Uploads

- Use the sidebar "Upload Image / Screenshot" to upload photos and screenshots.
- Uploaded images are saved to a local `uploads/` folder (created automatically).

## OCR / Image Analysis

For better OCR (text extraction) support, install the Tesseract engine and the Python wrapper:

- Debian/Ubuntu:

```bash
sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev
```

- macOS (Homebrew):

```bash
brew install tesseract
```

Then install the Python package (already added to `requirements.txt`):

```bash
pip install -r requirements.txt
```

After installing Tesseract, the app's "Extract text (OCR)" button will use `pytesseract` to extract text from uploaded images.

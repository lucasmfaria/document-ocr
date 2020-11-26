# document-ocr

This project includes a pipeline for OCR solutions. The steps are:
- Classification of documents
- Specific/General Preprocessing of images
- Optical Character Recognition (OCR)

## Classification

I use a VGG16 network to classify the images, using the Tensorflow framework.

## Preprocessing

This step consists in image preprocessing techniques, which result in improving the OCR step.

## OCR

I use the pytesseract module (python wrapper for the Tesseract-ocr solution).

Tesseract-ocr: https://github.com/tesseract-ocr/tesseract
Pytesseract: https://github.com/madmaze/pytesseract

## TODOs

- VGG16 train script
- Support for parameter optimization using scikit-learn API
- Support for document detection (improve OCR performance)
- Support for pytorch framework
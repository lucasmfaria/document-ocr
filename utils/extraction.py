from PIL import Image
import numpy as np
import pytesseract

def specific_pre_proc_ocr(image, class_name):
    '''
    Function to generate specific preprocessing steps for each document type
    The preprocessing steps should improve the performance of the OCR unit (Tesseract)
    :param image:
    :param class_name:
    :param ocr_tesseract:
    :return:
    '''
    img = image.copy()
    if class_name == 'X':
        pass
        #specific preprocess for document type X
    if class_name == 'Y':
        pass
        #specific preprocess for document type Y
    return img

def ocr(image, class_name=None):
    '''
    Function to perform OCR (Optical Character Recognition) on an image
    :param image:
    :param class_name:
    :return:
    '''
    img = Image.open(image)
    img = img.convert('L')
    img_np = np.array(img)

    #TODO - deal with document orientation

    if class_name:
        #specific OCR
        img_np = specific_pre_proc_ocr(img_np, class_name)
        text = pytesseract.image_to_string(img_np)
    else:
        #general OCR
        #simple preprocessing step:
        #img_np = simple_preprocess(img_np)
        text = pytesseract.image_to_string(img_np)

    return text
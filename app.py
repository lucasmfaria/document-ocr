import appconfig as config
from utils.classification import classify
from utils.extraction import ocr
from utils.utils import output_initialization

def process(image, process_types=['classification', 'ocr']):
    output = output_initialization()
    class_name = None
    if 'classification' in process_types:
        class_name = classify(image, config)
        output['class_name'] = class_name

    if 'ocr' in process_types:
        text = ocr(image, class_name)
        output['ocr'] = text

    return output
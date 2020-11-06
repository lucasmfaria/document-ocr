from PIL import Image
import tensorflow as tf
import numpy as np

def load_image(image_path, config):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((config.IMG_SIZE[0], config.IMG_SIZE[1]))

def load_model(config):
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    base_model = tf.keras.applications.vgg16.VGG16(weights=config.PRE_TRAINED_WEIGHTS_PATH,
                                                   include_top=config.INCLUDE_TOP,
                                                   pooling='max') #TODO - use the VGG16 structure without loading it's pre-trained weights

    inputs = tf.keras.Input(shape=(config.IMG_SIZE[0], config.IMG_SIZE[1], 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dense(config.N_HIDDEN, activation='relu')(x)
    outputs = tf.keras.layers.Dense(len(config.CLASS_NAMES), activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights(config.TRAINED_WEIGHTS_PATH)

    return model

def classify(image, config):
    #Load image if its a path from a directory:
    img = load_image(image)

    #TODO - load if its base64
    #TODO - load if its a numpy array or other objects

    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    model = load_model(config)

    img_np = np.array(img)
    img_tensor = tf.convert_to_tensor(preprocess_input(img_np))

    class_name = config.CLASS_NAMES[np.argmax(model.predict(tf.expand_dims(img_tensor, axis=0)))]

    return class_name
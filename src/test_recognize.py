import cv2
import numpy as np 

def recognize(image, model, color_labels):
    image = cv2.resize(image,  (224, 224))
    x = np.expand_dims(image, axis=0)

    my_color = model.predict(x, batch_size=1)
    my_color = color_labels[np.argmax(my_color)]

    return my_color
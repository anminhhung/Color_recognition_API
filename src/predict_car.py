import cv2
import numpy as np 

def predict(image, model, labels):
    image = cv2.resize(image, (150, 150))
    x = np.expand_dims(image, axis=0)

    result = model.predict(x, batch_size=1)
    result = labels[np.argmax(result)]

    return result
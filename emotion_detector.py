from keras.models import model_from_json
import numpy as np
import cv2


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


facec = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# def __init__(self, model_json_file, model_weights_file):
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

# load weights into the new model
loaded_model.load_weights("model_weights.h5")
loaded_model._make_predict_function()


def predict_emotion(img):
    preds = loaded_model.predict(img)
    return FacialExpressionModel.EMOTIONS_LIST[np.argmax(preds)]


def get_frame(img_data):
    gray_fr = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    if len(faces) < 1:
        pred = "No Face Detected"
    else:
        for (x, y, w, h) in faces:
            fc = gray_fr[y : y + h, x : x + w]
            roi = cv2.resize(fc, (48, 48))
            pred = predict_emotion(roi[np.newaxis, :, :, np.newaxis])
    return pred
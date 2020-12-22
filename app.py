import os
import cv2

# from model import FacialExpressionModel
import numpy as np
from keras.models import model_from_json
import numpy as np
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn
from starlette.templating import Jinja2Templates
import io
import time
from starlette.config import Config
from emotion_detector import *

# getting all the templets for the following dir.
templates = Jinja2Templates(directory="templates")


async def upload(request):
    if request.method == "POST":
        # Get the file from post request
        form = await request.form()
        f = form["image"].file
        image = io.BytesIO(f.read())
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        start = time.time()
        try:
            emotion = str(get_frame(img))
            end = time.time()
            context = {
                "Emotion": emotion,
                "Time_of_execution_in_seconds": str(round(end - start, 3)),
            }
        except:
            print("Invalied Input Image : Please enter the image with some Human faces")
            end = time.time()
            context = {
                "Invalied image": "Please enter the image with some Human face",
                "Time_of_execution_in_seconds": str(round(end - start, 3)),
            }
        return JSONResponse(context)
    return templates.TemplateResponse(
        "upload_emotionDetection.html", {"request": request, "data": ""}
    )


# All the routs of this websites
routes = [
    Route("/upload-image-DeepLobe-EmotionDetection", upload, methods=["GET", "POST"]),
]
# App congiguration.
app = Starlette(
    debug=True,
    routes=routes,
)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

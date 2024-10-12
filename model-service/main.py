from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

from loguru import logger
from PIL import Image
import time
import io

import torch
from torchvision import transforms
import numpy as np

from FCBFormer.Models import models

# GLOBALS
app = FastAPI(root_path="/")

# prep model
transform_input4test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((352, 352), antialias=True),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
model = models.FCBFormer()
state_dict = torch.load('./trained_weight/FCBFormer_Kvasir.pt', map_location=torch.device('cpu'))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.load_state_dict(state_dict["model_state_dict"])


@app.post("/segment")
async def segment(file: UploadFile = File(...)):

    # Read image from route
    request_object_content = await file.read()
    pil_image = Image.open(io.BytesIO(request_object_content))
    logger.info("pil_image: {}".format(pil_image))

    t = time.time()
    model.to(device)
    model.eval()
    t_load_model = time.time() - t

    t = time.time()
    image_tensor = transform_input4test(pil_image).unsqueeze(0).to(device)
    output = model(image_tensor)
    predicted_map = np.array(output.cpu().detach().numpy())
    predicted_map = np.squeeze(predicted_map)
    predicted_map = predicted_map > 0

    predicted_map_uint8 = (predicted_map * 255).astype(np.uint8)
    image = Image.fromarray(predicted_map_uint8, mode='L')  # 'L' mode for grayscale (black and white)
    t_predict = time.time() - t

    # Save the image to an in-memory file-like object
    t = time.time()
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    t_postprocess = time.time() - t

    logger.info("load model: {:.2f}s, predict: {:.2f}s, postprocess: {:.2f}s".format(t_load_model, t_predict, t_postprocess))

    return StreamingResponse(img_io, media_type="image/png")


# @app.post("/chat")
# def run_chat(text: str):
#     logger.info(f"User: {text}")
#     return {"response": "Hello World"}

# if __name__ == "__main__":
#     # print(1)
#     uvicorn.run(app, host="0.0.0.0", port=8000)
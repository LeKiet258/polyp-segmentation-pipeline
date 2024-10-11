from fastapi import FastAPI, File, UploadFile
from FCBFormer import predict
from loguru import logger
import uvicorn

app = FastAPI(root_path="/")

@app.post("/chat")
def run_chat(text: str):
    logger.info(f"User: {text}")
    return {"response": "Hello World"}

# @app.post("/ocr")
# async def predict(file: UploadFile = File(...)):

#     reader = easyocr.Reader(
#         ["vi", "en"],
#         gpu=False,
#         detect_network="craft",
#         model_storage_directory="./my_model",
#         download_enabled=False,
#     )
#     # Read image from route
#     request_object_content = await file.read()
#     pil_image = Image.open(BytesIO(request_object_content))
#     logger.info("pil_image")
#     logger.info(pil_image)

#     # Get the detection from EasyOCR
#     detection = reader.readtext(pil_image)

#     # Create the final result
#     result = {"bboxes": [], "texts": [], "probs": []}
#     for bbox, text, prob in detection:
#         # Convert a list of NumPy int elements to premitive numbers
#         bbox = np.array(bbox).tolist()
#         result["bboxes"].append(bbox)
#         result["texts"].append(text)
#         result["probs"].append(prob)

#     return result


if __name__ == "__main__":
    # print(1)
    uvicorn.run(app, host="0.0.0.0", port=8000)
"Mostly accurate but can miss some coins as only 16 images have been given to the model"
import os
from dotenv import load_dotenv

from inference_sdk import InferenceHTTPClient

load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in .env file")

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

# infer on a local image - this is an example image
result = CLIENT.infer("/Users/dhanika/Downloads/IMG_5569.jpg", model_id="cointrackertest-nuypl/2")

for prediction in result["predictions"]:
    if prediction["confidence"] > 0.7:
        print(prediction["class"], prediction["confidence"])
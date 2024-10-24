import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

REPLICATE_API_KEY = "Replace_with_your_api_key"
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"


# Define a request model for input prompts
class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "stability-ai/stable-diffusion"
    num_outputs: int = 1
    width: int = 512
    height: int = 512


@app.post("/generate-image/")
async def generate_image(request: ImageGenerationRequest):
    headers = {
        "Authorization": f"Token {REPLICATE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Payload to send to Replicate's API
    payload = {
        "version": "latest",
        "input": {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_outputs": request.num_outputs
        }
    }

    try:
        # Send the request to Replicate's API
        response = requests.post(
            f"{REPLICATE_API_URL}", json=payload, headers=headers)

        # Check if the response is OK
        if response.status_code == 201:
            data = response.json()
            return {"status": "success", "images": data["urls"]}
        else:
            raise HTTPException(
                status_code=response.status_code, detail=response.json())

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


# Home endpoint to test the API
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Image Generation App using Replicate!"}

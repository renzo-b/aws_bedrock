import boto3
import json
import base64
from PIL import Image
import io



promp_data = """
Make images of fruit
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "textToImageParams": {
        "text": promp_data
    },
    "taskType": "TEXT_IMAGE",
    "imageGenerationConfig": {
        "cfgScale": 8,
        "seed": 42,
        "quality": "standard",
        "width": 1024,
        "height": 1024,
        "numberOfImages": 3
    }
}


body = json.dumps(payload)
model_id = "amazon.titan-image-generator-v2:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
base64_image = response_body.get("images")[0]
base64_bytes = base64_image.encode('ascii')
image_bytes = base64.b64decode(base64_bytes)

image = Image.open(io.BytesIO(image_bytes))
image.show()
import boto3
import json

promp_data = """
Act as Shakespeare and write a poem on machine learning
"""

bedrock = boto3.client(service_name="bedrock-runtime")

payload = {
    "prompt":promp_data,
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p": 0.9
}

body = json.dumps(payload)
model_id = "meta.llama3-8b-instruct-v1:0"

response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]

print(response_text)
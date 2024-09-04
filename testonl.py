from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="CRqrGW810SkcQWx0u7Ns"
)

# infer on a local image
result = CLIENT.infer(R"C:\Users\students\Pictures\traffic\tra.png", model_id="mysearch/1")


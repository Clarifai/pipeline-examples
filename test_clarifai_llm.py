"""Quick script to verify Clarifai LLM connectivity.

Usage:
    export CLARIFAI_PAT=<your-pat>
    python test_clarifai_llm.py
"""

from clarifai.client.user import User

user = User(user_id="openai")
model = user.app(app_id="chat-completion").model(model_id="gpt-4o")

resp = model.predict_by_bytes(
    input_bytes=b"Say hello in 3 words",
    input_type="text",
    inference_params={"temperature": 0.1, "max_tokens": 50},
)

print(resp)

print(f"Response: {resp.outputs[0].data.text.raw}")

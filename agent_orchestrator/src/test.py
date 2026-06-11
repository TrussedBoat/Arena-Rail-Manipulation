import base64
import time
from openai import OpenAI

# 1. Configuration
MODEL_NAME = "Qwen3.6-35B"
IMAGE_PATH = "/home/homerobotics/Pictures/Screenshots/exp4-2.png"

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    timeout=120.0  # Generous timeout for large context + vision tasks
)

# 2. Encode local image to Base64
with open(IMAGE_PATH, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# 3. Execute request and track elapsed time
start_time = time.time()

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": "You are a robotic task orchestrator. For a given image and a task, you should give subtasks with format 'pick up the <object-name> and place it into the <object-name>.'",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "task: arrange the table"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ],
    temperature=0.2,
    max_tokens=4096,  # High token ceiling to accommodate the --reasoning overhead
)

elapsed_time = time.time() - start_time

# 4. Clean Output Extraction
answer = response.choices[0].message.content

print("--- ORCHESTRATION PLAN ---")
print(answer if answer else "No text returned. Check server logs for internal errors.")

print("\n--- PERFORMANCE METRICS ---")
print(f"Elapsed Time: {elapsed_time:.2f} seconds")
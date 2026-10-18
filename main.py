from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-5-5"
max_tokens=500
messages=[
        {
            "role": "user",
            "content": "What is quantum computing? Answer in one sentence"
        }
    ]

message = client.messages.create(
    model=model,
    max_tokens=max_tokens,
    messages=messages
)

print("Response Text: ", message.content[0].text)
print("\n---------------------------------\n")
print(message.to_json(indent=2))
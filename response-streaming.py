from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-5-5"

messages = []

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

add_user_message(messages, "Write a 1 sentence description of a fake database")

with client.messages.stream(
    model=model,
    max_tokens=500,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="\n")
    
    # Get the complete message for database storage
    final_message = stream.get_final_message()
    print("final_message: ", final_message.content[0].text)

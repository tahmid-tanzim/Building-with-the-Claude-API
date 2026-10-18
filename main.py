from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-5-5"
max_tokens=500
turn_index=1

# Start with an empty message list
messages = []

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    global turn_index
    print(f"\n----------------START ({turn_index})-----------------\n")
    print(messages)
    print(message.to_json(indent=2))
    print(f"\n----------------END ({turn_index})-----------------\n")
    turn_index += 1
    return message.content[0].text


# Add the initial user question
add_user_message(messages, "Define quantum computing in one sentence")

# Get Claude's response
answer = chat(messages)

# Add Claude's response to the conversation history
add_assistant_message(messages, answer)

# Add a follow-up question
add_user_message(messages, "Write another sentence")

# Get the follow-up response with full context
final_answer = chat(messages)

print("Response Text: ", final_answer)

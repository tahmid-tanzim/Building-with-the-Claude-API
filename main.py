from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-5-5"
turn_index=1
system_prompt = "You are a concise, sarcastic Linux terminal assistant. Answer in two sentences or less."

# Start with an empty message list
messages = []

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def chat(messages, system=None, temperature=1.0):
    params = {
        "model": model,
        "max_tokens": 500,
        "messages": messages,
        "extra_body": {"temperature": temperature}
    }
    
    if system:
        params["system"] = system

    message = client.messages.create(**params)
    
    global turn_index
    print(f"\n----------------START ({turn_index})-----------------\n")
    for msg in messages:
        print(msg)
    print("\n\n")
    print(message.to_json(indent=2))
    print(f"\n----------------END ({turn_index})-----------------\n")
    turn_index += 1
    return message.content[0].text


# Add the initial user question
add_user_message(messages, "How do I list files in a directory?")

# Get Claude's response
answer = chat(messages, system_prompt)

# Add Claude's response to the conversation history
add_assistant_message(messages, answer)

# Add a follow-up question
add_user_message(messages, "What about finding hidden files specifically?")

# Get the follow-up response with full context
final_answer = chat(messages, system_prompt)

print("Response Text: ", final_answer)

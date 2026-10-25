import json
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

client = Anthropic()
model = "claude-haiku-5-5"
turn_index = 1


def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})


def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})


def log_turn(messages, message):
    global turn_index
    print(f"\n==================== TURN {turn_index} START ====================\n")
    print("--- Messages ---")
    for msg in messages:
        print(msg)
    print("\n--- Response ---")
    print(message.to_json(indent=2))
    print(f"\n==================== TURN {turn_index} END ======================\n")
    turn_index += 1


def chat(messages, system=None, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 2000,
        "messages": messages,
        "extra_body": {"temperature": temperature}
    }

    if system:
        params["system"] = system

    if stop_sequences:
        params["stop_sequences"] = stop_sequences

    message = client.messages.create(**params)
    log_turn(messages, message)

    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "".join(text_blocks)


def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
  {
    "task": "Description of task",
  },
  ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code
* Respond with ONLY the JSON array, no other text before or after it

Please generate 3 objects.
"""

    messages = []
    add_user_message(messages, prompt)
    text = chat(messages)

    return json.loads(strip_code_fence(text))


def strip_code_fence(text):
    """Strips a markdown code fence (e.g. ```json ... ```) if the model adds one"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    return text


def run_prompt(test_case):
    """Merges the prompt and test case input, then returns the result"""
    prompt = f"""
Please solve the following task:

{test_case["task"]}
"""

    messages = []
    add_user_message(messages, prompt)
    return chat(messages)


def run_test_case(test_case):
    """Calls run_prompt, then grades the result"""
    output = run_prompt(test_case)

    # TODO - Grading
    score = 10

    return {
        "output": output,
        "test_case": test_case,
        "score": score
    }


def run_eval(dataset):
    """Loads the dataset and calls run_test_case with each case"""
    return [run_test_case(test_case) for test_case in dataset]


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def grade_by_model(test_case, output):
    # Create evaluation prompt
    eval_prompt = """
    You are an expert code reviewer. Evaluate this AI-generated solution.
    
    Task: {task}
    Solution: {solution}
    
    Provide your evaluation as a structured JSON object with:
    - "strengths": An array of 1-3 key strengths
    - "weaknesses": An array of 1-3 key areas for improvement  
    - "reasoning": A concise explanation of your assessment
    - "score": A number between 1-10
    """
    
    messages = []
    add_user_message(messages, eval_prompt)
    add_assistant_message(messages, "```json")

    eval_text = chat(messages, stop_sequences=["```"])
    return json.loads(eval_text)

DATASET_INPUT_PATH = "dataset-input.json"
DATASET_OUTPUT_PATH = "dataset-output.json"

if __name__ == "__main__":
    dataset = generate_dataset()
    save_json(DATASET_INPUT_PATH, dataset)

    dataset = load_json(DATASET_INPUT_PATH)
    results = run_eval(dataset)
    save_json(DATASET_OUTPUT_PATH, results)

    results = load_json(DATASET_OUTPUT_PATH)
    for index, result in enumerate(results, start=1):
        print(f"\n==================== RESULT {index} START ====================\n")
        print(result["output"])
        print(f"\n==================== RESULT {index} END ======================\n")

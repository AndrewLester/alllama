from llama_cpp import Llama, CreateChatCompletionStreamResponse
from typing import cast, Iterator

llm = Llama(
	model_path="./models/meta-llama-3-70B-instruct-IQ2_XS.gguf",
	n_gpu_layers=-1,
	n_ctx=4096,
	verbose=False,
)

message = input("Enter message: ")

print(flush=True)

output = cast(
	Iterator[CreateChatCompletionStreamResponse],
	llm.create_chat_completion(
		messages=[
			{
				"role": "system",
				"content": "You are a chat bot assistant. You will now begin receiving prompts from a single user.",
			},
			{"role": "user", "content": message},
		],
		stop=["<|eot_id|>"],
		stream=True,
	),
)

for chunk in output:
	if "content" in chunk["choices"][0]["delta"]:
		print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
print()

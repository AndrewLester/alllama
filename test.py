from typing import Iterator, cast

from llama_cpp import CreateChatCompletionStreamResponse

from llm import create_history, llm

history = create_history()

while True:
	message = input("Enter message:\n")
	print()

	history.append(
		{"role": "user", "content": message},
	)

	print("Assistant:", flush=True)

	output = cast(
		Iterator[CreateChatCompletionStreamResponse],
		llm.create_chat_completion(
			messages=history,
			stop=["<|eot_id|>"],
			stream=True,
		),
	)

	displayed_output = ""
	for chunk in output:
		delta = chunk["choices"][0]["delta"]
		if "content" in delta:
			content = cast(str, delta.get("content", ""))
			print(content, end="", flush=True)
			displayed_output += content
	print("\n")

	history.append({"role": "system", "content": displayed_output})

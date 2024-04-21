from typing import cast

from llama_cpp import ChatCompletionRequestMessage, Llama

llm = Llama(
	model_path="./models/meta-llama-3-70B-instruct-IQ2_XS.gguf",
	n_gpu_layers=-1,
	n_ctx=4096,
	verbose=False,
)

system_prompt = """
				You are a chat bot assistant. You will now begin receiving prompts from a single user.
				"""


def create_history():
	return cast(
		list[ChatCompletionRequestMessage],
		[
			{
				"role": "system",
				"content": system_prompt,
			},
		],
	)

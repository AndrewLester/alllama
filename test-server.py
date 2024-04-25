import openai

client = openai.OpenAI(
	base_url="https://llama-3-instruct-70b-iq2xs.andrewlester.net/v1",  # "http://<Your api-server IP>:port"
	api_key="KEY_HERE",
)

completion = client.chat.completions.create(
	model="gpt-3.5-turbo",
	messages=[
		{
			"role": "system",
			"content": "You are Llama 3, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
		},
		{"role": "user", "content": "Code a snake game in Python plz"},
	],
	stream=True,
	stop=["<|eot_id|>"],
)


for chunk in completion:
	print(chunk.choices[0].delta.content or "", end="", flush=True)
print()

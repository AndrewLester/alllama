import json
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Iterator, cast

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from llama_cpp import ChatCompletionRequestMessage, CreateChatCompletionStreamResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from llm import llm


@asynccontextmanager
async def lifespan(app: FastAPI):
	async with httpx.AsyncClient(timeout=None) as client:
		yield {"client": client}


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
	return "Pong"


class CompletionRequest(BaseModel):
	conversation: list[ChatCompletionRequestMessage]
	key: str


completion_lock = Lock()


@app.post("/completion")
def completion(completion_request: CompletionRequest):
	if completion_request.key != os.environ.get("API_SECRET"):
		return Response(status_code=401)

	completion_lock.acquire()

	output = cast(
		Iterator[CreateChatCompletionStreamResponse],
		llm.create_chat_completion(
			messages=completion_request.conversation,
			stop=["<|eot_id|>"],
			stream=True,
		),
	)

	content = (json.dumps(chunk) for chunk in output)

	def close():
		content.close()
		completion_lock.release()

	return StreamingResponse(
		content,
		media_type="text/event-stream",
		background=BackgroundTask(close),
	)


@app.get("/")
async def home(request: Request):
	conversation = [
		{"role": "system", "content": "You are a chat bot. Here is a user's message:"},
		{"role": "user", "content": "Write me code for a snake game plz"},
	]

	host = "http://127.0.0.1:8000"

	client = request.state.client
	req = client.build_request(
		"POST",
		f"{host}/completion",
		json={"conversation": conversation, "key": os.environ.get("API_SECRET")},
	)
	res = await client.send(req, stream=True)

	content = (
		json.loads(chunk)["choices"][0]["delta"].get("content", "")
		async for chunk in res.aiter_text()
	)

	return StreamingResponse(
		content, media_type="text/event-stream", background=BackgroundTask(res.aclose)
	)

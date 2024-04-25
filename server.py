import os
import signal
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import openai
from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask


@asynccontextmanager
async def lifespan(app: FastAPI):
	async with httpx.AsyncClient(timeout=None) as client:
		yield {"client": client}


app = FastAPI(lifespan=lifespan)

security = HTTPBearer()


@app.get("/ping")
def ping():
	return "Pong"


@app.get("/stop")
def stop():
	os.kill(os.getpid(), signal.SIGTERM)


@app.post("/v1/chat/completions")
async def completion(
	request: Request,
	credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
	body: Annotated[Any, Body()],
):
	if credentials.credentials != os.environ.get("API_SECRET"):
		return Response(status_code=401)

	client: httpx.AsyncClient = request.state.client
	req = client.build_request(
		"POST",
		"http://localhost:8080/v1/chat/completions",
		json=body,
	)
	res = await client.send(req, stream=True)

	return StreamingResponse(
		res.aiter_raw(),
		status_code=res.status_code,
		headers=res.headers,
		media_type="text/event-stream",
		background=BackgroundTask(res.aclose),
	)


@app.get("/")
def home():
	http_client = httpx.Client(timeout=None)

	client = openai.OpenAI(
		base_url="http://localhost:8000/v1",
		api_key=os.environ.get("API_SECRET"),
		http_client=http_client,
	)

	response = client.chat.completions.create(
		model="llama-3",
		messages=[
			{
				"role": "system",
				"content": "You are Llama 3, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests.",
			},
			{"role": "user", "content": "Write me code for a snake game plz"},
		],
		stream=True,
		stop=["<|eot_id|>"],
	)

	content = (chunk.choices[0].delta.content or "" for chunk in response)

	def close():
		response.close()
		http_client.close()

	return StreamingResponse(
		content,
		media_type="text/event-stream",
		background=BackgroundTask(close),
	)

import argparse
import multiprocessing
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
import openai
import uvicorn
from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask


@asynccontextmanager
async def lifespan(app: FastAPI):
	if "LLAMA_BASE_URL" not in os.environ:
		raise RuntimeError(
			"LLAMA_BASE_URL environment variable must be set to llama.cpp server base URL."
		)

	async with httpx.AsyncClient(
		timeout=None, base_url=os.environ["LLAMA_BASE_URL"]
	) as client:
		yield {"client": client}


server_status_condition = multiprocessing.Condition()
server_stop_event = multiprocessing.Event()

app = FastAPI(lifespan=lifespan, server_status_condition=server_status_condition)

security = HTTPBearer()


@app.get("/ping")
def ping():
	return {"inference_server_status": not server_stop_event.is_set()}


@app.get("/llama/stop")
def stop(secret: str):
	if secret != os.environ.get("CONTROL_SECRET"):
		return Response(status_code=401)

	server_stop_event.set()
	server_status_condition.acquire()
	server_status_condition.notify()
	server_status_condition.release()


@app.get("/llama/start")
def start(secret: str):
	if secret != os.environ.get("CONTROL_SECRET"):
		return Response(status_code=401)

	server_stop_event.clear()
	server_status_condition.acquire()
	server_status_condition.notify()
	server_status_condition.release()


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
		"/v1/chat/completions",
		json=body,
	)

	try:
		res = await client.send(req, stream=True)
	except httpx.ConnectError:
		# TODO: Start llama.cpp server
		return Response(
			status_code=504,
			content="Starting up inference server... please wait 1 minute.",
		)

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
		base_url=f"http://{host}:{port}/v1",
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


def run_uvicorn():
	uvicorn.run(
		app,
		host=host,
		port=port,
		log_level="info",
		env_file="secrets.env",
		reload=False,
		# workers=workers, unfortunately due to need to access multiprocessing Event, we can't use workers
	)


def run_proxy() -> multiprocessing.Process:
	process = multiprocessing.Process(target=run_uvicorn, name="alllama-uvicorn")
	process.start()
	return process


def run_server(
	server_program: str, model_file: str, gpu_layers: int
) -> subprocess.Popen:
	return subprocess.Popen(
		[
			server_program,
			"-m",
			model_file,
			"-c",
			"4096",
			"-ngl",
			str(gpu_layers),
			"-cb",
			"-np",
			"3",
			"--slots-endpoint-disable",  # Can't read user data from llama.cpp
		],
		stdout=subprocess.DEVNULL,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		prog="alllama",
		description="A proxy for a llama.cpp server",
	)
	parser.add_argument(
		"-s",
		"--server",
		dest="server",
		required=True,
	)
	parser.add_argument(
		"-m",
		"--model",
		dest="model",
		required=True,
	)
	parser.add_argument(
		"-ngl", "--n_gpu_layers", dest="gpu_layers", required=True, type=int
	)

	args = parser.parse_args()

	dev = os.environ.get("ENV") != "production"

	host = "localhost"
	port = 8001
	if not dev:
		port = 8000

	proxy_process = run_proxy()

	with multiprocessing.Manager() as manager:
		while True:
			print("Starting llama.cpp server...")
			server_process = run_server(args.server, args.model, args.gpu_layers)

			server_status_condition.acquire()
			server_status_condition.wait_for(lambda: server_stop_event.is_set())

			print("Stopping llama.cpp server...")
			server_process.terminate()
			server_process.wait()

			server_status_condition.acquire()
			server_status_condition.wait_for(lambda: not server_stop_event.is_set())

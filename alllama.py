import argparse
import asyncio
import multiprocessing
import os
import queue
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Annotated, Any, TypedDict, cast

import httpx
import openai
import uvicorn
from fastapi import BackgroundTasks, Body, Depends, FastAPI, Request
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


class AppExtra(TypedDict):
	server_stop_event: threading.Event
	server_status_condition: threading.Condition
	server_completion_event_message_queue: multiprocessing.Queue


def get_app_extra() -> AppExtra:
	return cast(AppExtra, app.extra)


app = FastAPI(lifespan=lifespan)

security = HTTPBearer()


@app.get("/ping")
def ping():
	return {
		"inference_server_status": not get_app_extra()["server_stop_event"].is_set()
	}


@app.get("/llama/stop")
def stop(secret: str):
	if secret != os.environ.get("CONTROL_SECRET"):
		return Response(status_code=401)

	extra = get_app_extra()

	extra["server_status_condition"].acquire()
	extra["server_stop_event"].set()
	extra["server_status_condition"].notify()
	extra["server_status_condition"].release()


@app.get("/llama/start")
def start(secret: str):
	if secret != os.environ.get("CONTROL_SECRET"):
		return Response(status_code=401)

	extra = get_app_extra()

	extra["server_status_condition"].acquire()
	extra["server_stop_event"].clear()
	extra["server_status_condition"].notify()
	extra["server_status_condition"].release()


@app.post("/v1/chat/completions")
async def completion(
	request: Request,
	credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
	body: Annotated[Any, Body()],
	background_tasks: BackgroundTasks,
):
	if credentials.credentials != os.environ.get("API_SECRET"):
		return Response(status_code=401)

	client: httpx.AsyncClient = request.state.client
	req = client.build_request(
		"POST",
		"/v1/chat/completions",
		json=body,
	)

	extra = get_app_extra()

	try:
		res = await client.send(req, stream=True)
	except httpx.ConnectError:
		print("Auto-scaling up")
		extra["server_status_condition"].acquire()
		extra["server_stop_event"].clear()
		extra["server_status_condition"].notify()
		extra["server_status_condition"].release()

		await asyncio.sleep(1)  # Give it a sec

		req = client.build_request(
			"POST",
			"/v1/chat/completions",
			json=body,
		)
		res = await client.send(req, stream=True)

	extra["server_completion_event_message_queue"].put_nowait("completion")

	async def stream_content():
		try:
			async for chunk in res.aiter_raw():
				yield chunk
		except httpx.RemoteProtocolError:
			# Inference server was stopped, llama.cpp server doesn't follow protocol when force stopped
			pass

	return StreamingResponse(
		stream_content(),
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

	try:
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
	except openai.InternalServerError as error:
		return Response(status_code=500, content=error.body)

	content = (chunk.choices[0].delta.content or "" for chunk in response)

	return StreamingResponse(
		content,
		media_type="text/event-stream",
		background=BackgroundTask(http_client.close),
	)


def auto_scale(server_completion_event_message_queue: multiprocessing.Queue):
	while True:
		try:
			server_completion_event_message_queue.get(timeout=10 * 60)
			print("msg received")
		except queue.Empty:
			if server_stop_event.is_set():
				continue

			print("Auto-scaling down")
			server_status_condition.acquire()
			server_stop_event.set()
			server_status_condition.notify()
			server_status_condition.release()


def run_uvicorn(
	server_stop_event: threading.Event,
	server_status_condition: threading.Condition,
	server_completion_event_message_queue: multiprocessing.Queue,
):
	app.extra = {
		"server_stop_event": server_stop_event,
		"server_status_condition": server_status_condition,
		"server_completion_event_message_queue": server_completion_event_message_queue,
	}

	uvicorn.run(
		app,
		host=host,
		port=port,
		log_level="info",
		env_file="secrets.env",
		reload=False,
		# workers=workers, unfortunately due to need to access multiprocessing Event, we can't use workers
	)


def run_proxy(
	server_stop_event: threading.Event,
	server_status_condition: threading.Condition,
	server_completion_event_message_queue: multiprocessing.Queue,
) -> multiprocessing.Process:
	process = multiprocessing.Process(
		target=run_uvicorn,
		args=(
			server_stop_event,
			server_status_condition,
			server_completion_event_message_queue,
		),
		daemon=True,
		name="alllama-uvicorn",
	)
	process.start()
	return process


def run_auto_scaler(
	server_completion_event_message_queue: multiprocessing.Queue,
) -> multiprocessing.Process:
	process = multiprocessing.Process(
		target=auto_scale,
		args=(server_completion_event_message_queue,),
		daemon=True,
		name="alllama-auto-scaler",
	)
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

	with multiprocessing.Manager() as manager:
		server_stop_event = manager.Event()
		server_status_condition = manager.Condition()
		server_completion_event_message_queue = cast(
			multiprocessing.Queue, manager.Queue()
		)

		# "Daemon" processes
		proxy_process = run_proxy(
			server_stop_event,
			server_status_condition,
			server_completion_event_message_queue,
		)
		auto_scale_process = run_auto_scaler(server_completion_event_message_queue)

		while True:
			print("Starting llama.cpp server...")
			server_process = run_server(args.server, args.model, args.gpu_layers)

			server_status_condition.acquire()
			server_status_condition.wait_for(lambda: server_stop_event.is_set())
			server_status_condition.release()

			print("Stopping llama.cpp server...")
			server_process.terminate()
			server_process.terminate()  # Send two terminates to close existing connections
			server_process.wait(timeout=5)

			server_status_condition.acquire()
			server_status_condition.wait_for(lambda: not server_stop_event.is_set())
			server_status_condition.release()

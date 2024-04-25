# llama-server

```
uvicorn server:app --env-file secrets.env
```

```
./server -m ../../llama-server/models/meta-llama-3-70B-instruct-IQ2_XS.gguf -c 4096 -ngl 81 -cb -np 3
```

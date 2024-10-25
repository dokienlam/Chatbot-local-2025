from ollama._client import Client
from ollama._types import GenerateResponse

_all_ = [
    'Client',
    'GenerateResponse',
    'generate',
]

_client = Client()

generate = _client.generate
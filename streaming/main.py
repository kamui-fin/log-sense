import asyncio
import websockets
import json
from datetime import datetime, timedelta
from pprint import pprint
from urllib.parse import quote

LOKI_ENDPOINT = "ws://loki.logsense.comfy.box/loki/api/v1/tail"
LIMIT = 10
query = quote('{host="master"}|drop filename')


async def tail_loki():
    async with websockets.connect(
        LOKI_ENDPOINT + f"?limit={LIMIT}&query={query}"
    ) as websocket:
        while True:
            logs = json.loads(await websocket.recv())
            logs = logs["streams"]
            pprint(logs)


asyncio.get_event_loop().run_until_complete(tail_loki())

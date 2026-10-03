#!/usr/bin/env python

import asyncio
import json
import websockets

async def send_messages():
    async with websockets.connect("ws://localhost:8001") as websocket:

        message_in = {
            "input": "./dataset/example1/", # path pro folder do vídeo
            "output": "output.txt" # path pra salvar o output
        }

        # envia pro server
        await websocket.send(json.dumps(message_in))

        # recebe a confirmação
        handshake = await websocket.recv()
        accepted = json.loads(handshake)
        if accepted['event'] == 'message-accepted':
            # recebe a predição
            message_out = await websocket.recv()
            output = json.loads(message_out)['output']
            print(f'Inference complete. Output in {output}')
        else:
            print('Server rejected the message.')

if __name__ == "__main__":
    asyncio.run(send_messages())

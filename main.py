from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Set
import asyncio

from counting import VehicleCounting

app = FastAPI()

class SubscriptionManager():
    def __init__(self):
        self.subscribers: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()

    def subscribe(self, websocket: WebSocket):
        self.subscribers.add(websocket)

    def unsubscribe(self, websocket: WebSocket):
        self.subscribers.discard(websocket)

    async def broadcast_to_subscribers(self, data: dict):
        for connection in self.subscribers:
            try:
                await connection.send_json(data)
            except Exception:
                self.subscribers.discard(connection)

manager = SubscriptionManager()

@app.on_event("startup")
async def start_vehicle_counter_stream():
    detector = VehicleCounting(show_window=False)

    async def stream_loop():
        last_counts = {}
        while True:
            if not manager.subscribers:
                await asyncio.sleep(0.5)
                continue

            counts = await asyncio.to_thread(detector.process_next_frame)

            if counts and counts != last_counts:
                last_counts = dict(counts)
                await manager.broadcast_to_subscribers(
                    {
                        "event": "vehicle_counted",
                        "data": counts
                    }
                )

            await asyncio.sleep(0.001)

    asyncio.create_task(stream_loop())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            payload = await websocket.receive_json()
            action = payload.get("action")

            if action == "subscribe":
                manager.subscribe(websocket)
                await websocket.send_json(
                    {
                        "status": "SUCCESS",
                        "message": "Berhasil subscribe ke vehicle counting"
                    }
                )
                print("FE berhasil subscribe!")
            elif action == "unsubscribe":
                manager.unsubscribe(websocket)
                await websocket.send_json(
                    {
                        "status": "SUCCESS",
                        "message": "Berhenti subscribe."
                    }
                )
                print("FE unsubscribe!")
    except WebSocketDisconnect:
        manager.unsubscribe(websocket)
"""
Uses the Gemini 2.0 multimodal live API and auto-restarts before the session limit.
Outputs text (to be converted into speech later).

This does not work, due to a bug in the API when using text outputs. I mean, it's experimental, so oh well.

The session has lasted longer than 2 minutes... I'm going to leave it and see how long it lasts.
The session stayed on for 10 minutes, and ended with this error in "while result := await self._receive()":
websockets.exceptions.ConnectionClosedError: received 1011 (internal error) Request trace id: cc7ab8ecf54c1a56, [ORIGINAL ERROR] RPC::DEADLINE_EXCEEDED: ; then sent 1011 (internal error) Request trace id: cc7ab8ecf54c1a56, [ORIGINAL ERROR] RPC::DEADLINE_EXCEEDED:
"""

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import wave
import pyaudio
import PIL.Image
import mss
import time
import numpy as np

import argparse

from google import genai
from scipy.signal import resample

from dotenv import load_dotenv

load_dotenv()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
# TIMEOUT_DURATION = 30  # Seconds before the session should switch
TIMEOUT_DURATION = 5

MODEL = "models/gemini-2.0-flash-exp"
TOOLS = [
    {"google_search": {}},
    {"code_execution": {}},
]

CONFIG = {
    "generation_config": {"response_modalities": ["TEXT"]},
    "system_instruction": "You are a helpful assistant talking in a voice call, where the user is potentially sharing their screen (your text output will be converted to speech, so don't use symbols or emojis).",
    "tools": TOOLS,
}

SUMMARY_PROMPT = "Please provide a detailed summary of our conversation so far, excluding this request. Do not include any follow up questions. Talk from the perspective of a third person, calling me the user and you the bot."


class AudioLoop:
    def __init__(self, screen_share: bool = True):
        self.client = genai.Client(http_options={"api_version": "v1alpha"})
        self.pya = pyaudio.PyAudio()

        self.screen_share = screen_share  # Whether to send screen frames
        
        self.out_queue = None  # Things to send to the bot (audio/screen/text)

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.is_listening = False  # TODO

        self.is_migrating = False  # Whether we're switching to a new session
        self.migrating_event = asyncio.Event()  # Event for migration
        
        self.summary = None
        self.start_time = time.time()

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "Message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):
        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)  # Sends 1 frame every second
            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # print(f"{time.time() - self.start_time:.3f}: Sending {msg['mime_type']}")
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        # Sends an audio snippet every ~0.06 seconds
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_text(self):
        "Background task to read from the websocket and write text output"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if text := response.text:
                    # print(text, end="")
                    print(text)
                    continue
            
            if self.is_migrating:
                # Summary has been outputted
                self.migrating_event.set()

    async def run(self):
        try:
            async with (
                self.client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.screen_share:
                    tg.create_task(self.get_screen())
                tg.create_task(self.receive_text())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


async def main():
    # Initialize and run the bot
    curr_session = AudioLoop(screen_share=True)  # Set screen_share as needed
    curr_task = asyncio.create_task(curr_session.run())

    while True:
        # Allow the session to run
        await asyncio.sleep(TIMEOUT_DURATION)

        # TODO Wait for bot to be inactive (not talking, not interrupted)

        # Create new session
        print("Creating new session...")
        new_session = AudioLoop(screen_share=True)
        new_task = asyncio.create_task(new_session.run())

        # Generate summary
        curr_session.is_migrating = True
        await curr_session.session.send(input=SUMMARY_PROMPT, end_of_turn=True)
        await curr_session.migrating_event.wait()
        print("Generated new summary...")

        # Feed summary to new session
        # while not curr_session.audio_summary_queue.empty():
        #     audio = curr_session.audio_summary_queue.get_nowait()
        #     await new_session.out_queue.put(audio)
        
        # Cancel the old session gracefully
        curr_task.cancel()
        try:
            await curr_task
            print("Old session stopped (exited).")
        except asyncio.CancelledError:
            print("Old session stopped (cancelled).")
        
        # Switch the session
        curr_session = new_session
        curr_task = new_task


if __name__ == "__main__":
    asyncio.run(main())

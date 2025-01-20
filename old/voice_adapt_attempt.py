"""Contains voice channel related functionality."""

import base64
import io
import sys
import traceback
import asyncio
import threading

import cv2
import wave
import pyaudio
import PIL.Image
import mss
import time

from google import genai

import asyncio
import os
import platform
from pathlib import Path
from xml.sax.saxutils import escape

import azure.cognitiveservices.speech as speechsdk
import discord
import numpy as np
from discord.ext import commands, voice_recv
from scipy.signal import decimate, resample

from discord_bot import bot
from utils import get_logger, truncate_length

# ===== GOOGLE GEMINI =====

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
TIMEOUT_DURATION = 30  # Seconds before the session should switch
# TIMEOUT_DURATION = 5

MODEL = "models/gemini-2.0-flash-exp"
TOOLS = [
    {"google_search": {}},
    {"code_execution": {}},
]

CONFIG = {
    "generation_config": {"response_modalities": ["AUDIO"]},
    # "system_instruction": "",
    "tools": TOOLS,
}

SUMMARY_PROMPT = "Please provide a very concise summary of our conversation before this message. Do not include any follow up questions. Talk from the perspective of the user, calling me 'I said' and you 'Google said'. Start with 'Here is a summary of our conversation so far'. End with 'Now comment on something you see me doing'."


def convert_audio_format(output_audio: bytes, input_rate: int, output_rate: int) -> bytes:
    """
    Converts the output audio format to match the input audio format.
    
    Parameters:
        output_audio (bytes): The raw PCM audio data at the output rate.
        input_rate (int): The target sample rate (e.g., 16kHz).
        output_rate (int): The source sample rate (e.g., 24kHz).
        
    Returns:
        bytes: The resampled raw PCM audio data at the input rate.
    """
    # Convert bytes to numpy array (16-bit little-endian PCM)
    output_audio_np = np.frombuffer(output_audio, dtype=np.int16)
    
    # Calculate the number of samples for the new rate
    num_samples = int(len(output_audio_np) * input_rate / output_rate)
    
    # Resample the audio
    resampled_audio = resample(output_audio_np, num_samples)
    # resampled_audio = resample_poly(output_audio_np, up=2, down=3)
    
    # Convert back to bytes
    return resampled_audio.astype(np.int16).tobytes()


async def save_pcm_to_wav(pcm_data: bytes, filename: str, sample_rate: int, channels: int, sample_width: int):
    """
    Saves a queue of PCM audio data to a WAV file.

    Parameters:
        queue (asyncio.Queue): Queue containing PCM audio data as bytes.
        filename (str): The output WAV file name.
        sample_rate (int): The sample rate of the audio (e.g., 16000).
        channels (int): Number of audio channels (e.g., 1 for mono).
        sample_width (int): Sample width in bytes (e.g., 2 for 16-bit audio).
    """
    with wave.open(filename, 'wb') as wav_file:
        # Set WAV file parameters
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        # Write PCM data to the WAV file
        wav_file.writeframes(pcm_data)
    
    print(f"Saved WAV file as {filename}")


class AudioLoop:
    def __init__(self, screen_share: bool = True):
        self.client = genai.Client(http_options={"api_version": "v1alpha"})
        self.pya = pyaudio.PyAudio()

        self.screen_share = screen_share  # Whether to send screen frames

        self.audio_in_queue = None  # Audio responses from the bot
        self.out_queue = None  # Things to send to the bot (audio/screen/text)

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

        self.is_listening = False
        self.is_outputting = False

        self.is_migrating = False  # Whether we're switching to a new session
        self.migrating_event = asyncio.Event()  # Event for migration

        self.audio_summary_queue = asyncio.Queue()
        self.num_frames = 0
        self.start_time = time.time()

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
            self.num_frames += 1
            # print(f"On frame {self.num_frames}")

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            if not self.is_migrating:  # Only send additional things if not migrating
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

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            include_silence = True
            had_data = False
            async for response in turn:
                if data := response.data:
                    had_data = True
                    self.is_outputting = True
                    self.is_listening = False
                    # print(f"{self.is_outputting=}")
                    # print(f"{self.is_listening=}")
                    if self.is_migrating:
                        # Save summary audio for the next session
                        self.audio_summary_queue.put_nowait(data)
                    else:
                        self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    include_silence = False
                    print(text, end="")
                    continue

                if response.server_content.interrupted:
                    # If you interrupt the model, it sends a turn_complete.
                    # For interruptions to work, we need to stop playback.
                    # So empty out the audio queue because it may have loaded
                    # much more audio than has played yet.
                    print("Interrupted")
                    had_data = False
                    include_silence = False
                    self.is_listening = True
                    # print(f"{self.is_listening=}")
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                    break
            
            if had_data and self.is_migrating:
                # Summary has been outputted
                self.migrating_event.set()
            elif include_silence:
                # Add silence at the end (fixes abrupt audio cutoff)
                silence_duration = 0.3  # Silence duration in seconds
                silence = b'\x00\x00' * int(RECEIVE_SAMPLE_RATE * silence_duration)
                self.audio_in_queue.put_nowait(silence)

    async def play_audio(self):
        stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)
            if self.audio_in_queue.empty():
                self.is_outputting = False
                # print(f"{self.is_outputting=}")

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

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)

curr_gemini_session = None


async def gemini_runner():
    global curr_gemini_session
    # Initialize and run the bot
    curr_session = AudioLoop(screen_share=True)  # Set screen_share as needed
    curr_gemini_session = curr_session
    curr_task = asyncio.create_task(curr_session.run())

    while True:
        # Allow the session to run (or exit on error)
        done, pending = await asyncio.wait(
            [curr_task], 
            timeout=TIMEOUT_DURATION, 
            return_when=asyncio.FIRST_COMPLETED
        )

        # TODO Wait for bot to be inactive (not talking, not interrupted)
        if curr_task in pending:
            pass

        # Create new session
        print("Creating new session...")
        new_session = AudioLoop(screen_share=True)
        new_task = asyncio.create_task(new_session.run())
        curr_gemini_session = new_session

        # Generate summary
        curr_session.is_migrating = True
        await curr_session.session.send(input=SUMMARY_PROMPT, end_of_turn=True)
        await curr_session.migrating_event.wait()
        print("Generated new summary...")

        # Get full audio PCM
        raw_audio_summary = []
        while not curr_session.audio_summary_queue.empty():
            raw_audio_summary.append(curr_session.audio_summary_queue.get_nowait())
        raw_audio_summary = b"".join(raw_audio_summary)

        # Convert summary to the correct PCM format
        print(len(raw_audio_summary))
        audio_summary = convert_audio_format(raw_audio_summary, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Feed summary to new session
        print("Converted")
        await save_pcm_to_wav(audio_summary, "summary.wav", SEND_SAMPLE_RATE, CHANNELS, 2)
        await new_session.out_queue.put({"data": audio_summary, "mime_type": "audio/pcm"})
        
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
    
    curr_gemini_session = None

if platform.system() == "Darwin":  # Manual load for Mac
    discord.opus.load_opus("libopus.0.dylib")

SPEECH_LOG_PATH = Path("local/speechsdk.log")
Path(SPEECH_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)

stt_conns: dict[int, bool] = {}  # Who is currently talking?


def process_voice_packet(user: discord.Member, data: voice_recv.VoiceData) -> None:
    """Process a voice packet from another user (?)."""
    TARGET_BITRATE = 16000
    try:
        packet: voice_recv.rtp.AudioPacket = data.packet
        channel: discord.VoiceChannel = user.voice.channel

        # Check if audio is loud enough to initiate STT
        power_level = packet.extension_data.get(voice_recv.ExtensionID.audio_power)
        if power_level:
            power_level = int.from_bytes(power_level, "big")
            power_level = 127 - (power_level & 127)
            if power_level < 80 and user.id not in stt_conns:
                return  # Assume frivolous noise

        if packet.is_silence():
            # Stop the recognizer (if present) and close the stream
            if user.id in stt_conns:
                del stt_conns[user.id]
            return

        if user.id not in stt_conns:
            stt_conns[user.id] = True
        else:
            stt_conns[user.id]

        # Get PCM data and resample to 16 kHz
        raw_pcm = np.frombuffer(data.pcm, dtype=np.int16)
        orig_bitrate = channel.bitrate
        # Check if the bitrate is a multiple of the target bitrate
        if orig_bitrate % TARGET_BITRATE == 0:
            resampled_pcm = decimate(raw_pcm, orig_bitrate // TARGET_BITRATE)
        else:
            # Fallback if not a multiple
            num_samples = int(len(raw_pcm) * (TARGET_BITRATE / channel.bitrate))
            resampled_pcm = resample(raw_pcm, num_samples)

        # Convert back to bytes and write to Gemini stream
        resampled_bytes = resampled_pcm.astype(np.int16).tobytes()
        curr_gemini_session.audio_in_queue.put_nowait(resampled_bytes)
    except Exception as e:
        logger.exception(f"Error processing voice packet from {user.display_name}: {e}")


@bot.hybrid_group(name="vc", fallback="join")
async def join_vc(ctx: commands.Context) -> None:
    """Tell Bob to join your voice channel."""
    if ctx.author.voice is None:
        await ctx.send("! but ur not in vc?")
        return
    elif ctx.voice_client is not None:
        await ctx.send("! im already in vc?")
        return
    channel: discord.VoiceChannel = ctx.author.voice.channel

    # Start Gemini
    gemini_task = asyncio.create_task(gemini_runner())

    # vc_histories[channel.id] = ManualHistory()
    vc: voice_recv.VoiceRecvClient = await channel.connect(cls=voice_recv.VoiceRecvClient)
    vc.listen(voice_recv.BasicSink(process_voice_packet))
    await ctx.send(
        "! hopping on :D\n\nTip: VC works best if you use push-to-talk. Speak without pausing, and let Bob finish."  # noqa: E501
    )

    # Handle messages and leave if VC is empty
    while vc.is_connected() and len(vc.channel.members) > 1:
        pass
    if vc.is_connected():
        await vc.disconnect()
    
    gemini_task.cancel()
    try:
        await gemini_task
    except asyncio.CancelledError:
        print("Gemini background task canceled.")


@join_vc.command(name="leave")
async def leave_vc(ctx: commands.Context) -> None:
    """Tell Bob to leave your voice channel."""
    if ctx.voice_client is None:
        await ctx.send("! im not in vc?")
        return
    # Remove VC history (if it exists)
    channel_id = ctx.voice_client.channel.id
    # if channel_id in vc_histories:
    #     del vc_histories[channel_id]
    await ctx.voice_client.disconnect()
    await ctx.send("! ok bye D:")

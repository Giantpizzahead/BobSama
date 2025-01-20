"""
Uses the Gemini 2.0 multimodal live API and auto-restarts before the session limit.
Use headphones to avoid echos being detected as speech / interrupting the bot.
"""

import asyncio
import base64
import io
import traceback
import random
from pynput import keyboard

import wave
import pyaudio
import PIL.Image
import mss
import time
import numpy as np
from threading import Thread

from google import genai
from scipy.signal import resample, resample_poly

from dotenv import load_dotenv

load_dotenv()




# PUSH_TO_TALK_KEY = keyboard.KeyCode.from_char("v")
PUSH_TO_TALK_KEY = keyboard.Key.alt_l
# PUSH_TO_TALK_KEY = None

CHECKIN_PROMPT = """Briefly make a playful or useful comment on what I am doing."""
CHECKIN_LOWER_BOUND = 45
CHECKIN_UPPER_BOUND = 90




FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
TIMEOUT_DURATION = 480  # Seconds before the session should switch

MODEL = "models/gemini-2.0-flash-exp"
TOOLS = [
    {"google_search": {}},
    {"code_execution": {}},
]

CONFIG = {
    "generation_config": {"response_modalities": ["AUDIO"]},
    "system_instruction": "You are a user named BobSama chatting in a private Discord voice chat. You are a rising junior majoring in computer science at MIT and a playful teen who is a witty, kind, and empathetic gamer (great at League of Legends, you play mid and main Zoe, also a fan of rhythm games). If the user asks you about your life/gameplay, you should roleplay as if you also play the games they do. Avoid rambling for long, don't repeat yourself, bring up banter sometimes to keep conversations going, and be aware of sarcastic comments. You can refuse to answer a very inappropriate message by saying 'Filtered' exactly. Talk in a lighthearted tone.",
    "tools": TOOLS,
}

SUMMARY_PROMPT = "Please provide a very concise summary of our conversation before this message. Do not include any follow up questions. Talk from the perspective of the user, calling me 'I said' and you 'Google said'. Start with 'Here is a summary of our conversation so far'. End with 'Now very briefly comment on what I am doing'."

should_listen = PUSH_TO_TALK_KEY is None


def on_press(evt):
    # print(evt, "pressed")
    global should_listen
    try:
        if evt == PUSH_TO_TALK_KEY and not should_listen:
            should_listen = True
            sound_thread = Thread(target=play_wav, args=("res/discord-ptt-on.wav",))
            sound_thread.start()
    except AttributeError:
        pass

def on_release(evt):
    # print(evt, "released")
    global should_listen
    try:
        if evt == PUSH_TO_TALK_KEY and should_listen:
            should_listen = False
            sound_thread = Thread(target=play_wav, args=("res/discord-ptt-off.wav",))
            sound_thread.start()
    except AttributeError:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


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


def play_wav(file_path):
    """
    Plays a WAV file.

    Parameters:
        file_path (str): Path to the WAV file.
    """
    try:
        # Open the WAV file
        with wave.open(file_path, 'rb') as wf:
            # Set up PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            # Read and play chunks
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)

            # Clean up
            stream.stop_stream()
            stream.close()
            p.terminate()

    except Exception as e:
        print(f"Error playing WAV file: {e}")


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

        self.user_is_talking = False
        self.bot_is_talking = False

        self.is_migrating = False  # Whether we're switching to a new session
        self.migrating_event = asyncio.Event()  # Event for migration

        self.audio_summary_queue = asyncio.Queue()
        self.num_frames = 0
        self.start_time = time.time()
        self.next_checkin = time.time() + random.randint(CHECKIN_LOWER_BOUND, CHECKIN_UPPER_BOUND)

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
            if PUSH_TO_TALK_KEY:
                self.user_is_talking = should_listen
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if not should_listen:  # Only capture audio when push-to-talk is active
                # Make all the data zeros
                data = b"\x00" * len(data)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

            # Check if it's time to send a check-in
            if time.time() > self.next_checkin:
                await self.session.send(input=CHECKIN_PROMPT, end_of_turn=True)
                self.next_checkin = time.time() + random.randint(CHECKIN_LOWER_BOUND, CHECKIN_UPPER_BOUND)
            elif self.user_is_talking or self.bot_is_talking:
                self.next_checkin = time.time() + random.randint(CHECKIN_LOWER_BOUND, CHECKIN_UPPER_BOUND)
                

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            include_silence = True
            had_data = False
            async for response in turn:
                if data := response.data:
                    had_data = True
                    if not self.bot_is_talking:
                        self.bot_is_talking = True
                        # print(f"{self.bot_is_talking=}")
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
                    print("Bot interrupted")
                    had_data = False
                    include_silence = False
                    if self.bot_is_talking:
                        self.bot_is_talking = False
                        # print(f"{self.bot_is_talking=}")
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
                if self.bot_is_talking:
                    self.bot_is_talking = False
                    # print(f"{self.bot_is_talking=}")

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


async def main():
    # Initialize and run the bot
    curr_session = AudioLoop(screen_share=True)  # Set screen_share as needed
    curr_task = asyncio.create_task(curr_session.run())
    print("BobSama is online!")

    while True:
        # Allow the session to run (or exit on error)
        done, pending = await asyncio.wait(
            [curr_task], 
            timeout=TIMEOUT_DURATION, 
            return_when=asyncio.FIRST_COMPLETED
        )

        # TODO Wait for bot to be inactive (not talking, not interrupted)
        if curr_task in pending:
            while curr_session.user_is_talking or curr_session.bot_is_talking:
                await asyncio.sleep(0.01)
        else:
            # Bot quit on its own
            print("Manual quit, exiting...")
            break

        # Create new session
        print("Creating new session...")
        new_session = AudioLoop(screen_share=True)
        new_task = asyncio.create_task(new_session.run())

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
        # print(len(raw_audio_summary))
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


if __name__ == "__main__":
    asyncio.run(main())

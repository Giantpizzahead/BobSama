"""
Example Gemini 2.0 Multimodal API usage
https://github.com/google-gemini/cookbook/blob/main/gemini-2/live_api_starter.ipynb
"""

import asyncio
import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(http_options= {'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"

config = {"generation_config": {"response_modalities": ["TEXT"]}}

async def main():
    async with client.aio.live.connect(model=model_id, config=config) as session:
        message = "Hello? Gemini are you there?"
        print("> ", message, "\n")
        await session.send(input=message, end_of_turn=True)

        # For text responses, When the model's turn is complete it breaks out of the loop.
        turn = session.receive()
        async for chunk in turn:
            if chunk.text is not None:
                print(f'- {chunk.text}')

if __name__ == "__main__":
    asyncio.run(main())

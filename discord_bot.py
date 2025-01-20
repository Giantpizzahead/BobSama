"""
Contains Discord bot functionality.
"""

"""Contains main Discord bot functionality."""

import asyncio
import json
import os
import random
import re
from enum import Enum
from typing import Optional

import discord
from discord.ext import commands
from utils import get_logger

logger = get_logger(__name__)


class BobSama(commands.Bot):
    """BobSama's Discord bot."""

    def __init__(self, *args, **kwargs):
        """Initialize the bot."""
        super().__init__(*args, **kwargs)


def init_bot() -> BobSama:
    """Initialize the bot."""
    intents: discord.Intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    return BobSama(command_prefix="!", help_command=None, intents=intents)


bot: BobSama = init_bot()


def run_bot() -> None:
    """Run the bot. Blocks until the bot is stopped."""
    token: Optional[str] = os.getenv("DISCORD_TOKEN")
    if token is None:
        raise ValueError("DISCORD_TOKEN environment variable is not set.")
    bot.run(token, log_handler=None)


@bot.event
async def on_ready() -> None:
    """Log when BobSama is online."""
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} commands.")
    except Exception:
        logger.exception("Error syncing commands")
    logger.info("BobSama is online!")

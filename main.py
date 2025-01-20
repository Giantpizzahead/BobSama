"""
Runs BobSama.
"""

import discord_bot
import voice

logger = discord_bot.get_logger(__name__)


def main() -> None:
    """Run the bot."""
    discord_bot.run_bot()
    logger.info("Stopping...")


if __name__ == "__main__":
    main()

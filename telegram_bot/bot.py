# ============================================================
# telegram_bot/bot.py — Main Telegram bot listener thread
# ============================================================
import logging
import threading

import shared_state
import config

logger = logging.getLogger(__name__)

_stop_event = threading.Event()

COMMANDS = {
    "/start", "/status", "/analysis", "/news",
    "/trades", "/memory", "/autoon", "/autooff",
    "/pause", "/resume", "/risk", "/health", "/help",
}


def start():
    """Entry point — called as a daemon thread from main.py."""
    shared_state.set_thread_status("telegram_bot", "running")
    logger.info("Telegram bot started.")

    try:
        from telegram.ext import (
            Application,
            CommandHandler,
            MessageHandler,
            filters,
        )
        from telegram import Update
        from telegram.ext import ContextTypes
        import asyncio

        app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()

        # Register all slash commands
        for cmd in [c.lstrip("/") for c in COMMANDS]:
            app.add_handler(CommandHandler(cmd, _make_command_handler(cmd)))

        # Register free-text handler
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _text_handler)
        )

        logger.info("Telegram polling started.")
        app.run_polling(stop_signals=None)

    except Exception as exc:
        logger.error("Telegram bot crashed: %s", exc, exc_info=True)
        shared_state.set_thread_status("telegram_bot", f"error: {exc}")


def stop():
    _stop_event.set()


def _make_command_handler(cmd: str):
    """Factory that creates an async handler for each slash command."""
    async def handler(update, context):
        if not _is_allowed(update):
            return
        from telegram_bot.commands import handle_command
        response = handle_command(f"/{cmd}", update.effective_user.id)
        try:
            await update.message.reply_text(response, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(response)
    return handler


async def _text_handler(update, context):
    """Handle free-text messages → route to Claude chat handler."""
    if not _is_allowed(update):
        return

    user_text = update.message.text.strip()
    user_id   = update.effective_user.id

    if not user_text:
        return

    # Show typing indicator
    await update.message.chat.send_action(action="typing")

    from telegram_bot.chat_handler import handle_message
    response = handle_message(user_text, user_id)

    # Send as plain text — avoids Markdown parse errors from AI responses
    # (Gemini output often contains special chars that break Telegram Markdown)
    if len(response) <= 4096:
        try:
            await update.message.reply_text(response, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(response)  # fallback: no parse mode
    else:
        for chunk in _split_message(response):
            try:
                await update.message.reply_text(chunk, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(chunk)


def _is_allowed(update) -> bool:
    """Whitelist check — only configured user IDs can use the bot."""
    uid = update.effective_user.id if update.effective_user else None
    if uid not in config.TELEGRAM_ALLOWED_UIDS:
        logger.warning("Blocked message from unauthorised user: %s", uid)
        return False
    return True


def _split_message(text: str, max_len: int = 4000):
    """Split a long message into chunks respecting Telegram limits."""
    lines  = text.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)
    return chunks

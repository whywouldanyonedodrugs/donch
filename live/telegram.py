# live/telegram.py
import asyncio
import logging
from typing import Optional

import aiohttp


LOG = logging.getLogger("live_trader.telegram")

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self._sess: Optional[aiohttp.ClientSession] = None
        self.offset: Optional[int] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._sess is None or self._sess.closed:
            timeout = aiohttp.ClientTimeout(total=40, connect=10, sock_read=35)
            self._sess = aiohttp.ClientSession(timeout=timeout)
        return self._sess

    async def _reset_session(self) -> None:
        if self._sess is not None:
            try:
                await self._sess.close()
            except Exception:
                pass
        self._sess = None

    async def _req(self, method: str, **params):
        retries = 3
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                sess = await self._ensure_session()
                async with sess.post(f"{self.base}/{method}", json=params) as r:
                    r.raise_for_status()
                    data = await r.json()
                    if not isinstance(data, dict):
                        raise RuntimeError(f"telegram invalid payload type: {type(data)!r}")
                    return data
            except asyncio.CancelledError:
                raise
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionResetError,
                OSError,
                RuntimeError,
            ) as e:
                last_err = e
                await self._reset_session()
                if attempt < retries:
                    await asyncio.sleep(min(5.0, 0.5 * (2 ** (attempt - 1))))
                    continue
                raise
        raise RuntimeError(f"telegram request failed after retries: {last_err!r}")

    async def send(self, text: str):
        try:
            data = await self._req("sendMessage", chat_id=self.chat_id, text=text)
            if data.get("ok") is not True:
                LOG.warning("Telegram sendMessage not ok: %s", data)
                return False
            return True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            LOG.warning("Telegram send failed: %s", e)
            return False

    async def poll_cmds(self):
        data = await self._req("getUpdates", offset=self.offset, timeout=25, limit=20)
        for upd in data.get("result", []):
            self.offset = upd["update_id"] + 1
            if (m := upd.get("message")) and (txt := m.get("text")):
                yield txt.strip()

    async def close(self):
        await self._reset_session()

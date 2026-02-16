import aiohttp
from typing import Optional

class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self._sess: Optional[aiohttp.ClientSession] = None
        self.offset: Optional[int] = None

    async def _req(self, method: str, **params):
        if self._sess is None:
            self._sess = aiohttp.ClientSession()
        async with self._sess.post(f"{self.base}/{method}", json=params) as r:
            return await r.json()

    async def send(self, text: str):
        await self._req("sendMessage", chat_id=self.chat_id, text=text)

    async def poll_cmds(self):
        data = await self._req("getUpdates", offset=self.offset, timeout=0, limit=20)
        for upd in data.get("result", []):
            self.offset = upd["update_id"] + 1
            if (m := upd.get("message")) and (txt := m.get("text")):
                yield txt.strip()

    async def close(self):
        if self._sess:
            await self._sess.close()

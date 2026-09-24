import httpx

async def test():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        resp = await client.post("https://lite.duckduckgo.com/lite/", data={"q": "capital of france"})
        with open("ddg.html", "w", encoding="utf-8") as f:
            f.write(resp.text)
        print("Written to ddg.html")

import asyncio
asyncio.run(test())

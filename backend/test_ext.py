import asyncio
from app.services.retrieval.document_extractor import fetch_and_extract_article

async def test():
    url = "https://en.wikipedia.org/wiki/France"
    res = await fetch_and_extract_article(url)
    print("Title:", repr(res.get('title')))
    print("Author:", repr(res.get('author')))
    print("Date:", repr(res.get('date')))
    print("Text length:", len(res.get('text', '')))
    print("Text preview:", repr(res.get('text', '')[:200]))

asyncio.run(test())

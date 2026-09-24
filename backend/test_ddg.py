import asyncio
from app.services.search.ddg_provider import DDGSearchProvider

async def test():
    provider = DDGSearchProvider()
    results = await provider.search("capital of france 2", num_results=3)
    for i, r in enumerate(results):
        print(f"Result {i}:")
        print(f"  Title: {r.get('title')}")
        print(f"  URL: {r.get('url')}")
        print(f"  Snippet: {r.get('snippet')}")
        print()

asyncio.run(test())

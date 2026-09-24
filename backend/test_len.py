import asyncio
from app.services.search.ddg_provider import DDGSearchProvider
async def test():
    provider = DDGSearchProvider()
    results = await provider.search("Delhi India Gate shark road rainfall spot swim", num_results=10)
    print(f"Got {len(results)} results")
    for r in results:
        print(r['url'])
asyncio.run(test())

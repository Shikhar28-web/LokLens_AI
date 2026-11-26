import asyncio
import httpx
import json
import time

async def test_image_api():
    print("1. Creating image submission...")
    
    image_path = r"D:\Projects\Political_lens\Mumbai Monsoon Flood News Bulletin.png"
    
    # Use a very long timeout since forensics analysis + web search can take 60+ seconds
    timeout = httpx.Timeout(120.0, connect=10.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(image_path, "rb") as f:
            files = {"image": ("test_image.png", f, "image/png")}
            res = await client.post("http://localhost:8000/api/submissions", files=files)
            
        sub_id = res.json()["id"]
        print(f"Created submission: {sub_id}")
        
        print("2. Triggering analysis pipeline...")
        await client.post(f"http://localhost:8000/api/submissions/{sub_id}/analyze")
        
        print("3. Polling for completion (this can take up to 2 minutes)...")
        for i in range(60):  # poll for up to 2 minutes
            await asyncio.sleep(3)  # wait before each poll
            try:
                status_res = await client.get(f"http://localhost:8000/api/submissions/{sub_id}")
                status = status_res.json()["status"]
                print(f"  [{i*3}s] Status: {status}")
                if status in ["complete", "error"]:
                    break
            except httpx.ReadTimeout:
                print(f"  [{i*3}s] Server busy (still processing)...")
                continue
            
        print("4. Fetching final report...")
        report_res = await client.get(f"http://localhost:8000/api/submissions/{sub_id}/report")
        print("\n--- FINAL REPORT ---")
        print(json.dumps(report_res.json(), indent=2))

if __name__ == "__main__":
    asyncio.run(test_image_api())


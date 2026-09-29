import asyncio
import httpx
import sys
import time

async def test_api():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("1. Creating submission...")
        res = await client.post("http://localhost:8000/api/submissions", data={"text": "The Prime Minister announced that the government had created 10 crore new jobs in India since 2014."})
        
        if res.status_code != 201:
            print("Failed to create submission:", res.text)
            return
            
        sub = res.json()
        sub_id = sub["id"]
        print(f"Created submission: {sub_id}")
        
        print("2. Triggering analysis pipeline...")
        res = await client.post(f"http://localhost:8000/api/submissions/{sub_id}/analyze")
        if res.status_code != 200:
            print("Failed to trigger analysis:", res.text)
            return
            
        print("3. Polling for completion...")
        for _ in range(30):
            res = await client.get(f"http://localhost:8000/api/submissions/{sub_id}")
            status = res.json()["status"]
            print(f"Status: {status}")
            if status == "complete" or status == "error":
                break
            await asyncio.sleep(2)
            
        if status == "error":
            print("Pipeline failed:", res.json().get("error_message"))
            return
            
        print("4. Fetching final report...")
        res = await client.get(f"http://localhost:8000/api/submissions/{sub_id}/report")
        print("\n--- FINAL REPORT ---")
        import json
        print(json.dumps(res.json(), indent=2))

if __name__ == "__main__":
    asyncio.run(test_api())

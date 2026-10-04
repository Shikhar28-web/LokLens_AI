import requests
import json
import sys
import time

BASE_URL = 'http://localhost:8000/api'
payload = {"text": "Microsoft completes acquisition of Activision Blizzard for $69 billion in 2023."}
resp = requests.post(f'{BASE_URL}/submissions', data=payload)
if resp.status_code != 201:
    print('Failed to upload:', resp.text)
    sys.exit(1)
sub_id = resp.json()['id']
print('Submission ID:', sub_id)

resp = requests.post(f'{BASE_URL}/submissions/{sub_id}/analyze')
if resp.status_code != 200:
    print('Failed to trigger:', resp.text)
    sys.exit(1)

print('Polling for completion...')
while True:
    resp = requests.get(f'{BASE_URL}/submissions/{sub_id}')
    status = resp.json()['status']
    print('Status:', status)
    if status in ('complete', 'error'):
        break
    time.sleep(2)

print('Fetching report...')
resp = requests.get(f'{BASE_URL}/submissions/{sub_id}/report')
print(json.dumps(resp.json(), indent=2))

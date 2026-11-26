import time
import requests
import sys

BASE_URL = 'http://localhost:8000/api'
IMAGE_PATH = r'data\uploads\1c891d50-2101-43e7-a4e7-0c8f771c1447.jpeg'

print('Uploading image...')
with open(IMAGE_PATH, 'rb') as f:
    files = {'image': ('test.jpeg', f, 'image/jpeg')}
    resp = requests.post(f'{BASE_URL}/submissions', files=files)
    if resp.status_code != 201:
        print('Failed to upload:', resp.text)
        sys.exit(1)
    
sub_id = resp.json()['id']
print('Submission ID:', sub_id)

print('Triggering analysis...')
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
import json
print(json.dumps(resp.json(), indent=2))

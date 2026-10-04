from PIL import Image, ImageDraw, ImageFont
import requests
import json
import time
import sys

# Generate Image
img = Image.new('RGB', (800, 200), color = (255, 255, 255))
d = ImageDraw.Draw(img)
# Simple text, PIL default font is tiny but we can just use default
text = "Microsoft completes acquisition of Activision Blizzard for $69 billion in 2023."
d.text((10,10), text, fill=(0,0,0))
img.save('real_news_test.png')
print("Image generated!")

BASE_URL = 'http://localhost:8000/api'
IMAGE_PATH = 'real_news_test.png'

print('Uploading image...')
with open(IMAGE_PATH, 'rb') as f:
    files = {'image': ('real_news_test.png', f, 'image/png')}
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
print(json.dumps(resp.json(), indent=2))

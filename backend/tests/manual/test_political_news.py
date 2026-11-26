"""
Test political news classification via the LokLens REST API.
Verifies both genuine and fabricated political news samples.
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

samples = [
    {
        "category": "Fabricated Political News (Sensational Misinformation)",
        "title": "Shocking Video Exposes Stolen Ballots Found In Secret Dumpster!",
        "text": "BREAKING BOMBSHELL: Video proof circulating online shows election workers burning thousands of ballots marked for the opposition party in an unmarked alley dumpster behind counting facilities!"
    },
    {
        "category": "Fabricated Political News (Conspiracy Theory)",
        "title": "Secret Cabal Met In Switzerland To Plan Global Government Takeover",
        "text": "Whistleblower leaks classified agenda detailing how unelected elites orchestrated pandemic to dissolve national sovereignties and enforce martial law."
    },
    {
        "category": "Fabricated Political News (Fabricated Endorsement)",
        "title": "BREAKING: Pope Francis Shocks World With Trump Endorsement",
        "text": "News outlets around the world report that Pope Francis made an unprecedented move to endorse Donald Trump for President, citing secret Vatican communique defending conservative faith."
    },
    {
        "category": "Genuine Political News (Reuters Wire)",
        "title": "Senate Confirms Federal Judge in Bipartisan Vote",
        "text": "WASHINGTON (Reuters) - The United States Senate confirmed the nomination of the federal appellate judge in a 65-34 bipartisan vote following committee hearings."
    },
    {
        "category": "Genuine Political News (Official Government Release)",
        "title": "Treasury Department Releases Monthly Deficit Report",
        "text": "The federal government ran a budget deficit of $129 billion in February, according to monthly statement data published by the Treasury Department on Monday."
    }
]

print("=" * 70)
print("LOKLENS AI - POLITICAL NEWS CLASSIFICATION VERIFICATION")
print("=" * 70)

for s in samples:
    resp = client.post("/api/ml/classify-news", json={"text": s["text"], "title": s["title"]})
    print(f"[{s['category']}]")
    print(f"Headline: {s['title']}")
    print(f"HTTP Status: {resp.status_code}")
    data = resp.json()
    print(f"Predicted Label : {data.get('label')}")
    print(f"Confidence      : {data.get('confidence'):.4f}")
    print(f"Model Engine    : {data.get('model_type')}")
    print(f"All Scores      : {data.get('all_scores')}")
    print(f"Status Note     : {data.get('note')}")
    print("-" * 70)

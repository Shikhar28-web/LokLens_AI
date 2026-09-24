import asyncio
from app.database import AsyncSessionLocal
from app.models.submission import Submission
from app.models.verdict import Verdict
from sqlalchemy import select
from datetime import datetime, timezone

async def test():
    async with AsyncSessionLocal() as session:
        sub_id = '51e0deb2-3ff7-4e83-b88a-24e9214ddb84'
        result = await session.execute(select(Submission).where(Submission.id == sub_id))
        sub = result.scalar_one_or_none()
        print('Found sub:', sub.id if sub else None)
        sub.status = 'complete'
        sub.completed_at = datetime.now(timezone.utc)
        
        stub_verdict = Verdict(
            submission_id=sub_id,
            claim_verdict='INSUFFICIENT_EVIDENCE',
            claim_confidence=0.0,
            image_verdict='INCONCLUSIVE',
            image_confidence=0.0,
            overall_status='PIPELINE_STUB',
            independent_source_count=0,
            explanation_json={'summary': 'test'},
            limitations_json=['test'],
            score_components_json={'test': 'test'},
        )
        session.add(stub_verdict)
        try:
            await session.commit()
            print('Commit successful')
        except Exception as e:
            print('Commit failed:', e)

asyncio.run(test())

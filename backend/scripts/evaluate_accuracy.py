"""
LokLens AI — Manual Accuracy Evaluation Runner
Runs a benchmark suite of known Ground-Truth Claims and Images,
computes the Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.

v2 improvements:
- Uses the improved evidence_classifier v2 (negation-aware, expanded keywords)
- Snippet-level CONTRADICTS signal also scanned with the updated keyword list
- Falls back to FAKE rather than REAL for ambiguous results when snippets match
  known fact-check contradiction phrases
"""

import sys
import os
import asyncio

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.image_forensics.ela_analyzer import perform_ela
from app.services.image_forensics.frequency_analyzer import analyze_frequency
from app.services.image_forensics.noise_analyzer import analyze_noise
from app.services.image_forensics.copy_move_detector import detect_copy_move
from app.services.image_forensics.ai_score_estimator import estimate_ai_likelihood

from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.claim_extractor import extract_claims
from app.services.nlp.query_generator import generate_queries
from app.services.search.ddg_provider import DDGSearchProvider
from app.services.retrieval.document_extractor import fetch_and_extract_article
from app.services.retrieval.ranker import rank_and_extract_evidence
from app.services.nlp.evidence_classifier import (
    classify_evidence, CONTRADICTION_KEYWORDS, SUPPORT_KEYWORDS,
    _keyword_present_without_negation
)

# ── Comprehensive Realistic Benchmark Dataset (20 Diverse Ground-Truth Claims) ─
# Includes clear factual claims, viral hoaxes, subtle temporal distortions,
# and nuanced policy claims to prevent overfitting and reflect real-world performance.

BENCHMARK_CLAIMS = [

    # ==================================================
    # GROUND TRUTH: REAL (15 Claims)
    # ==================================================

    {
        "claim": "In September 2022, NASA successfully carried out its Double Asteroid Redirection Test (DART) mission by deliberately crashing a spacecraft into Dimorphos, a small asteroid orbiting Didymos. The experiment successfully changed Dimorphos' orbital period and demonstrated that a spacecraft could potentially be used to alter the trajectory of a threatening asteroid.",
        "ground_truth": "REAL"
    },
    {
        "claim": "In August 2023, India's Chandrayaan-3 mission successfully achieved a soft landing near the lunar south polar region. With this achievement, India became the fourth country in the world to successfully land a spacecraft softly on the Moon, following the United States, the Soviet Union and China. The mission also deployed the Pragyan rover to conduct experiments on the lunar surface.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The European Union officially brought its Artificial Intelligence Act into force on August 1, 2024. The legislation introduced a risk-based regulatory framework that categorizes AI applications according to their potential risks to individuals and society. It includes restrictions on certain prohibited AI practices and establishes obligations for providers of high-risk AI systems and general-purpose AI models.",
        "ground_truth": "REAL"
    },
    {
        "claim": "In January 2024, Japan successfully landed its Smart Lander for Investigating Moon (SLIM) spacecraft on the lunar surface. This achievement made Japan the fifth country to successfully achieve a soft landing on the Moon. The mission was designed to demonstrate highly precise landing technology and improve the accuracy of future lunar exploration missions.",
        "ground_truth": "REAL"
    },
    {
        "claim": "On May 5, 2023, the World Health Organization announced that COVID-19 no longer constituted a Public Health Emergency of International Concern. The decision followed improvements in global surveillance, vaccination coverage and the management of the disease. However, the WHO clarified that the announcement did not mean COVID-19 had completely disappeared or stopped being a global health concern.",
        "ground_truth": "REAL"
    },
    {
        "claim": "India's 2024 general election resulted in the Bharatiya Janata Party winning 240 seats in the Lok Sabha, fewer than the 272 seats required for an independent majority. However, the National Democratic Alliance secured enough seats to form the government, and Narendra Modi began his third consecutive term as Prime Minister after taking the oath of office in June 2024.",
        "ground_truth": "REAL"
    },
    {
        "claim": "NASA's OSIRIS-REx spacecraft successfully returned to Earth in September 2023 carrying samples collected from asteroid Bennu. The mission had originally traveled to the near-Earth asteroid to study its composition and gather material that could provide scientists with information about the early formation of the solar system. The returned samples were recovered in Utah and transported for scientific analysis.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The United Nations General Assembly declared 2025 the International Year of Quantum Science and Technology to recognize the contributions of quantum physics and mark approximately one hundred years since the development of modern quantum mechanics. The initiative aimed to increase public awareness, encourage scientific research and promote international cooperation in quantum science and its technological applications.",
        "ground_truth": "REAL"
    },
    {
        "claim": "During the COP28 climate conference held in Dubai in December 2023, participating countries adopted a global agreement that called for transitioning away from fossil fuels in energy systems. The agreement was considered significant because it explicitly addressed the role of fossil fuels in global warming, although it did not establish a legally binding requirement for every country to eliminate fossil fuel consumption by a specific deadline.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The Reserve Bank of India launched its retail Central Bank Digital Currency pilot project in December 2022. The digital rupee, also known as the e-Rupee, was introduced as a digital form of central bank money that could be used for retail transactions through participating banks. The pilot initially involved selected cities and participating financial institutions before being expanded further.",
        "ground_truth": "REAL"
    },
    {
        "claim": "Apple officially introduced its Vision Pro mixed-reality headset in the United States on February 2, 2024. The device combines augmented reality and virtual reality capabilities through a spatial computing interface that allows users to interact with digital applications in their physical surroundings. It also introduced features such as eye tracking, hand gestures and voice commands for controlling applications.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The 2024 Nobel Prize in Physics was awarded jointly to John J. Hopfield and Geoffrey Hinton for foundational discoveries and inventions that enabled machine learning using artificial neural networks. Their work contributed to the development of computational methods inspired by the human brain and helped establish important foundations for modern artificial intelligence technologies.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The European Space Agency launched its Jupiter Icy Moons Explorer mission, commonly known as JUICE, in April 2023. The spacecraft was designed to travel to Jupiter and conduct detailed observations of its icy moons, particularly Ganymede, Callisto and Europa. Its scientific objectives include investigating the moons' subsurface oceans and evaluating their potential habitability.",
        "ground_truth": "REAL"
    },
    {
        "claim": "India hosted the G20 Leaders' Summit in New Delhi in September 2023 under its presidency with the theme One Earth, One Family, One Future. During the summit, member countries and invited participants adopted the New Delhi Leaders' Declaration, which addressed international cooperation on economic development, climate change, digital public infrastructure and geopolitical challenges.",
        "ground_truth": "REAL"
    },
    {
        "claim": "The European Space Agency launched its Euclid space telescope in July 2023 to investigate the mysteries of dark matter and dark energy. The mission was designed to map the large-scale structure of the universe by observing billions of galaxies across cosmic history, helping scientists better understand how the universe has expanded and how its structure has evolved over time.",
        "ground_truth": "REAL"
    },


    # ==================================================
    # GROUND TRUTH: FAKE / CONTRADICTED (15 Claims)
    # ==================================================

    {
        "claim": "NASA's Double Asteroid Redirection Test mission successfully destroyed the asteroid Dimorphos completely after deliberately crashing a spacecraft into it in September 2022. Scientists confirmed that the asteroid was broken into thousands of pieces and that the mission permanently eliminated the possibility of any future asteroid collision with Earth.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "India's Chandrayaan-3 mission successfully landed directly on the exact geographic South Pole of the Moon in August 2023, becoming the first spacecraft in human history to reach that precise location. ISRO confirmed that the landing site was located at the Moon's southernmost point and that the Pragyan rover immediately began exploring permanently shadowed craters.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The European Union's Artificial Intelligence Act officially prohibited the development, training and commercial use of all generative artificial intelligence models across its member countries when it entered into force in August 2024. The legislation required technology companies to shut down existing AI chatbots and obtain special government permission before developing any new machine learning system.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "Japan's SLIM spacecraft successfully completed its lunar landing mission in January 2024 and subsequently continued its journey to Mars, where it became the first Japanese spacecraft to land on the Martian surface. The Japanese space agency confirmed that the mission had been designed to demonstrate landing technologies for both lunar and Martian exploration.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The World Health Organization officially announced the complete eradication of COVID-19 worldwide on May 5, 2023, after observing a decline in infection rates and hospital admissions. The organization confirmed that the coronavirus was no longer circulating among humans and that all countries could permanently discontinue vaccination and disease surveillance programmes.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The Bharatiya Janata Party secured 303 seats independently in India's 2024 Lok Sabha general election, allowing it to form the central government without requiring support from any coalition partner. The Election Commission confirmed that the party had improved its performance compared with the 2019 election and secured another single-party majority.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "NASA's OSIRIS-REx mission successfully collected rock and soil samples from the surface of Mars and returned them to Earth in September 2023. The spacecraft was originally developed to investigate the Martian atmosphere and collect material from the planet's surface to help scientists determine whether ancient life existed on Mars.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The United Nations General Assembly officially declared 2025 as the International Year of Artificial General Intelligence and announced that all member countries must establish national AGI development agencies. The resolution also required governments to complete the development of human-level artificial intelligence systems before the end of the year.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The COP28 climate conference in Dubai resulted in a legally binding international agreement requiring every participating country to completely stop using coal, petroleum and natural gas by December 2030. The agreement established automatic financial penalties for countries that continued operating fossil fuel-based power plants after the deadline.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The Reserve Bank of India introduced its first retail Central Bank Digital Currency pilot project in 2018 alongside the initial launch of the Unified Payments Interface. The project immediately made digital rupee transactions available to every Indian citizen through all commercial banks without requiring any special wallet or banking application.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "Apple launched its Vision Pro mixed-reality headset simultaneously in every country worldwide on January 1, 2024. The company announced that the device was immediately available through all international Apple stores and that its initial launch included India, China, Japan, the United States and every European Union member country.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The 2024 Nobel Prize in Physics was jointly awarded to Sam Altman and Elon Musk for their groundbreaking contributions to artificial intelligence, neural networks and the development of large language models. The Nobel Committee stated that their commercial achievements had transformed scientific research and established the foundation of modern machine learning.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "India's first hydrogen-powered passenger train officially began commercial operations between Delhi and Mumbai in January 2024. The Ministry of Railways announced that the train was powered entirely by domestically manufactured hydrogen fuel cells and had replaced conventional diesel locomotives on several major railway routes.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "The European Space Agency's JUICE spacecraft successfully entered orbit around Jupiter in April 2023 immediately after its launch from Earth. Scientists confirmed that the spacecraft completed its journey to the Jupiter system within a few hours and began transmitting detailed images of Europa and Ganymede during the same month.",
        "ground_truth": "FAKE"
    },
    {
        "claim": "India hosted the 2023 G20 Leaders' Summit in Mumbai instead of New Delhi, where participating countries adopted a final agreement known as the Mumbai Consensus. The declaration established India as the permanent headquarters of the G20 organization and transferred the group's administrative operations from its previous international offices.",
        "ground_truth": "FAKE"
    }
]


# Additional contradiction signal words (broader than CONTRADICTION_KEYWORDS for snippet scanning)
_SNIPPET_CONTRA_SIGNALS = set(CONTRADICTION_KEYWORDS) | {
    "hoax", "false", "fake", "fact check", "busted", "debunk", "misleading",
    "no such", "viral hoax", "viral fake", "rumour", "rumor", "propaganda",
    "unverified", "not confirmed", "wrong", "incorrect"
}

async def evaluate_claim(item):
    claim_text = item["claim"]
    expected = item["ground_truth"]

    # 1. Clean
    cleaned = normalize_text(claim_text)

    # 2. Search — use 3 queries now (core + fact-check + Indian fact-checker)
    provider = DDGSearchProvider()
    queries = generate_queries(cleaned)[:3]

    results = []
    for q in queries:
        try:
            res = await provider.search(q, num_results=3)
            if res:
                results.extend(res)
        except Exception:
            pass

    if not results:
        return {"claim": claim_text, "expected": expected, "predicted": "UNVERIFIED", "correct": False}

    has_support = False
    has_contra = False

    for r in results[:5]:
        url = r.get("url", "")
        try:
            doc = await fetch_and_extract_article(
                url,
                fallback_title=r.get("title", ""),
                fallback_snippet=r.get("snippet", "")
            )
            text = (doc.get("text", "") if doc else "") or r.get("snippet", "")
            if len(text) > 30:
                evs = rank_and_extract_evidence(claim_text, text, top_n=2)
                for ev in evs:
                    cls = classify_evidence(claim_text, ev["sentence"])
                    if cls["relationship"] == "SUPPORTS":
                        has_support = True
                    elif cls["relationship"] == "CONTRADICTS":
                        has_contra = True
        except Exception:
            continue

    # Evidence Stance Evaluation:
    # A robust, non-overfit model differentiates between pure support, pure contradiction,
    # and genuinely mixed/disputed context (MISLEADING/UNVERIFIED).
    if has_contra and not has_support:
        predicted = "FAKE"
    elif has_support and not has_contra:
        predicted = "REAL"
    elif has_contra and has_support:
        # Mixed evidence detected across sources -> classified as MISLEADING
        predicted = "MISLEADING"
    else:
        # Fallback snippet scan: scan indexed titles and descriptions
        snips = " ".join([r.get("snippet", "").lower() + " " + r.get("title", "").lower() for r in results])
        contra_found, _ = _keyword_present_without_negation(snips, list(_SNIPPET_CONTRA_SIGNALS))
        support_found, _ = _keyword_present_without_negation(snips, SUPPORT_KEYWORDS)

        if contra_found and not support_found:
            predicted = "FAKE"
        elif support_found and not contra_found:
            predicted = "REAL"
        elif contra_found and support_found:
            predicted = "MISLEADING"
        else:
            predicted = "UNVERIFIED"

    correct = (expected == "REAL" and predicted == "REAL") or \
              (expected == "FAKE" and predicted in ("FAKE", "CONTRADICTED", "MISLEADING"))
    return {
        "claim": claim_text,
        "expected": expected,
        "predicted": predicted,
        "correct": correct
    }

async def run_benchmark():
    print("=" * 70)
    print(" LokLens AI — Live Accuracy & Precision/Recall Benchmark Evaluator v2")
    print("=" * 70)
    print(f"Total benchmark claims: {len(BENCHMARK_CLAIMS)}")
    print("Testing against live web fact-checking sources...\n")

    tp = 0  # Expected REAL, Predicted REAL
    tn = 0  # Expected FAKE, Predicted FAKE
    fp = 0  # Expected FAKE, Predicted REAL
    fn = 0  # Expected REAL, Predicted FAKE/UNVERIFIED

    results = []
    for i, item in enumerate(BENCHMARK_CLAIMS, 1):
        print(f"[{i}/{len(BENCHMARK_CLAIMS)}] Evaluating: {item['claim'][:60]}...")
        res = await evaluate_claim(item)
        results.append(res)

        status_sym = "✅" if res["correct"] else "❌"
        print(f"      Expected: {res['expected']:<4} | Predicted: {res['predicted']:<10} | {status_sym}")

        # In fact-checking benchmarks, identifying a false claim as FAKE, CONTRADICTED,
        # or MISLEADING correctly flags the misinformation (True Negative).
        if res["expected"] == "REAL" and res["predicted"] == "REAL":
            tp += 1
        elif res["expected"] == "FAKE" and res["predicted"] in ("FAKE", "CONTRADICTED", "MISLEADING"):
            tn += 1
        elif res["expected"] == "FAKE" and res["predicted"] == "REAL":
            fp += 1
        else:
            fn += 1

    total = len(BENCHMARK_CLAIMS)
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 70)
    print(" BENCHMARK ACCURACY REPORT")
    print("=" * 70)
    print(f"True Positives  (TP) : {tp:<4} (Correctly verified REAL claims)")
    print(f"True Negatives  (TN) : {tn:<4} (Correctly debunked FAKE claims)")
    print(f"False Positives (FP) : {fp:<4} (Fake claims wrongly marked Real)")
    print(f"False Negatives (FN) : {fn:<4} (Real claims marked Fake/Unverified)")
    print("-" * 70)
    print(f"Accuracy  : {accuracy  * 100:.1f}%")
    print(f"Precision : {precision * 100:.1f}%")
    print(f"Recall    : {recall    * 100:.1f}%")
    print(f"F1-Score  : {f1        * 100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_benchmark())


import asyncio
import sys
from rich.console import Console
from rich.panel import Panel

from app.services.preprocessing.text_preprocessor import normalize_text
from app.services.nlp.ner import extract_entities
from app.services.nlp.claim_extractor import extract_claims
from app.services.nlp.query_generator import generate_queries
from app.services.search.ddg_provider import DDGSearchProvider
from app.services.retrieval.document_extractor import fetch_and_extract_article

console = Console()

async def run_test(input_text: str):
    console.print(Panel(f"[bold cyan]Raw Input:[/bold cyan]\n{input_text}"))
    
    # 1. Preprocessing
    console.print("\n[bold yellow]1. Text Preprocessing...[/bold yellow]")
    cleaned_text = normalize_text(input_text)
    console.print(f"Cleaned Text: [green]{cleaned_text}[/green]")
    
    # 2. Extract Claims
    console.print("\n[bold yellow]2. Extracting Claims...[/bold yellow]")
    claims = extract_claims(cleaned_text)
    if not claims:
        console.print("[red]No verifiable claims found in the text![/red]")
        return
        
    for i, claim in enumerate(claims, 1):
        console.print(f"Claim {i}: [green]{claim}[/green]")
        
    # For this test, we'll just take the first claim
    target_claim = claims[0]
    
    # 3. Named Entity Recognition
    console.print(f"\n[bold yellow]3. Extracting Entities for Claim 1...[/bold yellow]")
    entities = extract_entities(target_claim)
    console.print(entities)
    
    # 4. Query Generation
    console.print("\n[bold yellow]4. Generating Search Queries...[/bold yellow]")
    queries = generate_queries(target_claim, entities)
    for q in queries:
        console.print(f" - [cyan]{q}[/cyan]")
        
    # 5. Search DuckDuckGo (Fallback loop)
    console.print(f"\n[bold yellow]5. Searching DuckDuckGo...[/bold yellow]")
    provider = DDGSearchProvider()
    results = []
    
    for query in queries:
        console.print(f"Searching for: '{query}'...")
        results = await provider.search(query, num_results=5)
        if results:
            break
        console.print("[yellow]No results found, trying next query...[/yellow]")
    
    if not results:
        console.print("[red]No search results found![/red]")
        return
        
    for i, res in enumerate(results, 1):
        console.print(f"\n[bold cyan]Result {i}: {res['title']}[/bold cyan]")
        console.print(f"URL: {res['url']}")
        
    # 6. Fetch Document(s)
    console.print(f"\n[bold yellow]6. Fetching and Extracting Documents...[/bold yellow]")
    
    combined_text = ""
    success_count = 0
    
    for res in results:
        target_url = res['url']
        console.print(f"Attempting to fetch {target_url} ...")
        doc = await fetch_and_extract_article(target_url)
        
        # Only accept documents that have a reasonable amount of text (ignore JS walls)
        if doc and len(doc['text']) > 150:
            console.print(f"[green]Success! Extracted {len(doc['text'])} chars.[/green]")
            combined_text += f"\n\nSource: {target_url}\n{doc['text']}"
            success_count += 1
        else:
            console.print("[yellow]Failed to extract meaningful text (JS wall, video, or blocked).[/yellow]")
            
    if success_count == 0:
        console.print("[red]Failed to extract meaningful text from ANY search results![/red]")
        return
        
    console.print(f"\n[bold green]Successfully extracted {len(combined_text)} characters across {success_count} sources.[/bold green]")
    
    # 7. Extract Evidence & Classify (Phase 7 - NLI)
    console.print(f"\n[bold yellow]7. Scoring & Classifying Evidence (Phase 7 - NLI)...[/bold yellow]")
    from app.services.retrieval.ranker import rank_and_extract_evidence
    from app.services.nlp.evidence_classifier import classify_evidence
    
    matrix = []
    has_supports = False
    has_contradicts = False
    has_allegation = False
    
    # We will test all atomic claims against the evidence found
    for claim in claims:
        evidence = rank_and_extract_evidence(claim, combined_text, top_n=3)
        for ev in evidence:
            raw_score = round(ev["combined_score"], 2)
            
            # Classify using our NLI model
            classification = classify_evidence(claim, ev["sentence"])
            rel = classification["relationship"]
            is_alleg = classification["is_allegation"]
            conf = round(classification["confidence"] * 100, 1)
            
            status = "UNVERIFIED"
            if rel == "SUPPORTS":
                status = "SUPPORTED"
                has_supports = True
            elif rel == "CONTRADICTS":
                status = "CONTRADICTED"
                has_contradicts = True
            elif rel == "ATTRIBUTED_CLAIM":
                status = "ALLEGATION_ONLY"
                has_allegation = True
                
            matrix.append({
                "claim": claim,
                "evidence": ev["sentence"],
                "relationship": rel,
                "status": status,
                "retrieval_score": raw_score,
                "nli_conf": conf
            })
            
    if not matrix:
        console.print("[red]No relevant evidence found in the document![/red]")
        return
        
    # 8. Claim-vs-Evidence Matrix
    console.print("\n[bold yellow]8. Claim-vs-Evidence Matrix[/bold yellow]")
    from rich.table import Table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Claim")
    table.add_column("Evidence")
    table.add_column("Relationship")
    table.add_column("Status")
    table.add_column("Retr. Score")
    
    for row in matrix:
        table.add_row(
            row["claim"][:50] + "...", 
            row["evidence"][:100] + "...", 
            f"[cyan]{row['relationship']}[/cyan] ({row['nli_conf']}%)", 
            f"[bold {'green' if row['status'] == 'SUPPORTED' else 'red' if row['status'] == 'CONTRADICTED' else 'yellow'}]{row['status']}[/]", 
            str(row['retrieval_score'])
        )
    console.print(table)
    
    # 9. Final Report
    console.print("\n[bold yellow]9. Final Structured Report[/bold yellow]")
    console.print("[bold]Established Facts:[/bold]")
    if has_supports:
        console.print(" - The retrieved evidence supports parts of the claim.")
    else:
        console.print(" - No independent proof found to support the core claims.")
        
    console.print("\n[bold]Disputed / Contradicted:[/bold]")
    if has_contradicts:
        console.print(" - Evidence was found that contradicts the claim.")
    else:
        console.print(" - No direct contradictions were found.")
        
    console.print("\n[bold]Allegations / Unverified:[/bold]")
    if has_allegation:
        console.print(" - Portions of the claim are purely political allegations or accusations without established proof in the retrieved text.")
    else:
        console.print(" - None.")
        
    console.print("\n[italic]Note: High retrieval score indicates topic relevance, not proof. NLI confidence indicates semantic relationship.[/italic]")    
if __name__ == "__main__":
    test_text = (
        "Hello there! Did you know that the sky is green? "
        "Also, Elon Musk said Tesla will build a factory in Texas by 2025, costing $5 billion. "
        "I think it's going to be huge."
    )
    
    if len(sys.argv) > 1:
        test_text = sys.argv[1]
        
    asyncio.run(run_test(test_text))

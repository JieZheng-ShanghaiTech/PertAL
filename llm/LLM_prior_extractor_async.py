"""
Asynchronous LLM prior extractor.

This script scores candidate genes for a given cell line using an OpenAI-compatible
Chat Completions endpoint and writes a CSV of structured JSON outputs.

Security note:
  Do NOT hardcode API keys or private endpoints in source code. Configure via
  environment variables instead:

    - OPENAI_API_KEY   (required)
    - OPENAI_BASE_URL  (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL     (optional, default: gpt-4.1-mini)

Optional dataset I/O overrides:
    - DATASET_NAME, CELL_LINE, SEED, OUTPUT_DIR, GENE_LIST_FILE
"""

import asyncio
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
from openai import AsyncOpenAI

# --- Experimental context (override via env vars if desired) ---
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
CELL_LINE = os.getenv("CELL_LINE", "K562")
DATASET_NAME = os.getenv("DATASET_NAME", "Replogle")
SEED = int(os.getenv("SEED", "4"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "result_3_11"))
OUTPUT_FILE = OUTPUT_DIR / f"{DATASET_NAME}_{MODEL_NAME}_{CELL_LINE}_gene_priors_{SEED}.csv"
GENE_LIST_FILE = Path(os.getenv("GENE_LIST_FILE", f"{DATASET_NAME}_{CELL_LINE}_genes.txt"))


def build_async_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Refusing to run with hardcoded credentials."
        )
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60)


def load_candidate_genes(path: Path) -> List[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


    
def generate_structured_prompt(gene_name: str, cell_context: str):
    system_prompt = (
        "You are a computational systems biology expert with 20 years of experience in "
        "CRISPR screening and gene regulatory networks. Your task is to evaluate the potential "
        "biological significance and information value of a specific gene perturbation based on "
        "published literature and biological knowledge bases. You MUST output ONLY a strict JSON object."
    )
    user_prompt = f"""
    Please strictly evaluate the gene "{gene_name}" in the "{cell_context}" cell line.
    You MUST output ONLY a valid JSON object, with EXACTLY the following 3 numeric fields:
    {{
    "hubness_score": <integer from 0 to 10>,
    "phenotype_impact_score": <integer from 0 to 10>,
    "knowledge_scarcity_score": <integer from 0 to 10>
    }}
    Do NOT output any explanations, comments, or extra keys.
    The output must be directly parseable by json.loads.
    """
    return system_prompt, user_prompt



async def get_llm_score_async(gene_name: str, cell_context: str, client: AsyncOpenAI) -> Dict:
    system_prompt, user_prompt = generate_structured_prompt(gene_name, cell_context)
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        content = None
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            content = response.choices[0].message.content
            result_json = json.loads(content)
            required_keys = ['hubness_score', 'phenotype_impact_score', 'knowledge_scarcity_score']
            if all(key in result_json for key in required_keys):
                result_json['gene_name'] = gene_name
                result_json['error'] = None
                return result_json
            else:
                raise ValueError("JSON output is incomplete.")
        except Exception as e:
            print(f"[{gene_name}] Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if content is not None:
                print("Raw LLM output:", content)
            await asyncio.sleep(1.5)
    return {
        "gene_name": gene_name,
        "hubness_score": None,
        "phenotype_impact_score": None,
        "knowledge_scarcity_score": None,
        "error": "Failed after retries",
    }
async def batch_query_async(genes: List[str], cell_context: str, client: AsyncOpenAI, concurrency: int = 8) -> pd.DataFrame:
    """
    Concurrent async querying with a semaphore-based concurrency limit.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    async def sem_task(gene):
        async with semaphore:
            print(f"Querying {gene}...")
            result = await get_llm_score_async(gene, cell_context, client)

            return result
    tasks = [asyncio.create_task(sem_task(gene)) for gene in genes]
    for coro in asyncio.as_completed(tasks):
        res = await coro
        results.append(res)
    return pd.DataFrame(results)



if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = build_async_client()
    candidate_genes = load_candidate_genes(GENE_LIST_FILE)

    print(f"Starting async batch query for {len(candidate_genes)} genes using {MODEL_NAME}...")
    df_results = asyncio.run(batch_query_async(candidate_genes, CELL_LINE, client, concurrency=8))
    if "error" not in df_results.columns:
        df_results["error"] = None
    score_columns = ['hubness_score', 'phenotype_impact_score', 'knowledge_scarcity_score']
    df_results[score_columns] = df_results[score_columns].fillna(0)
    df_results['V_llm_raw'] = df_results[score_columns].sum(axis=1)
    df_results.to_csv(str(OUTPUT_FILE), index=False)
    success_count = len(df_results) - df_results['error'].notna().sum()
    error_count = df_results['error'].notna().sum()
    print(f"\n--- TASK COMPLETE ---")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Successfully queried {success_count} genes.")
    print(f"Failed {error_count} genes.")

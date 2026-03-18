"""
Asynchronous LLM prior judge/auditor.

This script reads a CSV produced by the prior-extractor, then asks an LLM to
audit/revise the three prior scores for biological plausibility.

Security note:
  Do NOT hardcode API keys or private endpoints in source code. Configure via
  environment variables instead:

    - OPENAI_API_KEY   (required)
    - OPENAI_BASE_URL  (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL_JUDGE (optional, default: OPENAI_MODEL or gpt-4.1-mini)
"""

import asyncio
import csv
import json
import os
import httpx
import pandas as pd
from typing import Dict, List
from openai import AsyncOpenAI

# --- Runtime config (override via env vars if desired) ---
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("OPENAI_MODEL_JUDGE", os.getenv("OPENAI_MODEL", "gpt-5.2"))

CELL_LINE = os.getenv("CELL_LINE", "K562")
DATASET = os.getenv("DATASET_NAME", "Replogle")
INPUT_FILE = os.getenv(
    "INPUT_FILE", f"{DATASET}_gpt-4.1-mini_{CELL_LINE}_gene_priors.csv"
)
OUTPUT_FILE = os.getenv("OUTPUT_FILE", f"{DATASET}_{MODEL}_{CELL_LINE}_judge_results.csv")
CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))  # flush to disk every N results


def build_async_client() -> AsyncOpenAI:
    if not API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Refusing to run with hardcoded credentials."
        )
    return AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

# column order for incremental CSV
COLUMNS = [
    "gene_name",
    "hubness_score",
    "phenotype_impact_score",
    "knowledge_scarcity_score",
    "V_llm_raw",
    "hubness_rationality",
    "phenotype_rationality",
    "scarcity_rationality",
    "rationality_mean",
    "hubness_revised",
    "phenotype_revised",
    "scarcity_revised",
    "V_llm_revised",
    "hubness_delta",
    "phenotype_delta",
    "scarcity_delta",
    "V_delta",
    "error",
]


async def probe(url: str, key: str, retries: int = 5, gap: float = 3.0) -> bool:
    """GET {base_url}/models with Bearer token, require HTTP 200."""
    endpoint = f"{url}/models"
    headers = {"Authorization": f"Bearer {key}"}
    for i in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10) as h:
                r = await h.get(endpoint, headers=headers)
            if r.status_code == 200:
                print(f"[probe] {endpoint} -> 200 OK")
                return True
            print(f"[probe] attempt {i + 1}/{retries}: status {r.status_code}")
        except Exception as e:
            print(f"[probe] attempt {i + 1}/{retries}: {e}")
        await asyncio.sleep(gap)
    return False


def build_prompt(gene: str, cell: str, scores: Dict) -> tuple[str, str]:
    system = (
        "You are a senior computational biologist specializing in CRISPR screens "
        "and gene regulatory networks. Your task is to audit LLM-generated gene "
        "prior scores for biological plausibility. You MUST output ONLY a strict JSON object."
    )
    user = f"""
A previous LLM scored gene "{gene}" in the "{cell}" cell line as follows:
- hubness_score: {scores["hubness_score"]}  (0-10, how central/hub-like the gene is in regulatory networks)
- phenotype_impact_score: {scores["phenotype_impact_score"]}  (0-10, expected phenotypic impact upon perturbation)
- knowledge_scarcity_score: {scores["knowledge_scarcity_score"]}  (0-10, how poorly characterized the gene is)

Evaluate whether each score is biologically reasonable for this gene in {cell} cells.
Output ONLY a valid JSON with EXACTLY these 9 fields:
{{
  "hubness_rationality": <integer 0-10, how reasonable the original hubness_score is>,
  "phenotype_rationality": <integer 0-10, how reasonable the original phenotype_impact_score is>,
  "scarcity_rationality": <integer 0-10, how reasonable the original knowledge_scarcity_score is>,
  "hubness_revised": <integer 0-10, your corrected hubness_score>,
  "phenotype_revised": <integer 0-10, your corrected phenotype_impact_score>,
  "scarcity_revised": <integer 0-10, your corrected knowledge_scarcity_score>,
  "hubness_delta": <integer, hubness_revised minus original hubness_score>,
  "phenotype_delta": <integer, phenotype_revised minus original phenotype_impact_score>,
  "scarcity_delta": <integer, scarcity_revised minus original knowledge_scarcity_score>
}}
Do NOT output any explanation, markdown, or extra keys.
"""
    return system, user


async def judge(gene: str, cell: str, scores: Dict, client: AsyncOpenAI) -> Dict:
    system, user = build_prompt(gene, cell, scores)
    for attempt in range(3):
        raw = None
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = resp.choices[0].message.content
            parsed = json.loads(raw)
            required = [
                "hubness_rationality",
                "phenotype_rationality",
                "scarcity_rationality",
                "hubness_revised",
                "phenotype_revised",
                "scarcity_revised",
                "hubness_delta",
                "phenotype_delta",
                "scarcity_delta",
            ]
            if all(k in parsed for k in required):
                parsed["gene_name"] = gene
                parsed["error"] = None
                return parsed
            raise ValueError("Incomplete JSON keys")
        except Exception as e:
            print(f"[{gene}] attempt {attempt + 1}/3: {e}")
            if raw:
                print(f"  raw: {raw[:200]}")
            await asyncio.sleep(1.5)
    return {
        "gene_name": gene,
        "error": "Failed after retries",
        **{
            k: None
            for k in [
                "hubness_rationality",
                "phenotype_rationality",
                "scarcity_rationality",
                "hubness_revised",
                "phenotype_revised",
                "scarcity_revised",
                "hubness_delta",
                "phenotype_delta",
                "scarcity_delta",
            ]
        },
    }


def enrich(row: Dict, result: Dict) -> Dict:
    """Merge original scores with judge output into a flat row."""
    rev_cols = ["hubness_revised", "phenotype_revised", "scarcity_revised"]
    rat_cols = ["hubness_rationality", "phenotype_rationality", "scarcity_rationality"]
    out = {
        "gene_name": row["gene_name"],
        "hubness_score": row["hubness_score"],
        "phenotype_impact_score": row["phenotype_impact_score"],
        "knowledge_scarcity_score": row["knowledge_scarcity_score"],
        "V_llm_raw": row["V_llm_raw"],
        **{k: result.get(k) for k in rat_cols},
        **{k: result.get(k) for k in rev_cols},
        **{
            k: result.get(k)
            for k in ["hubness_delta", "phenotype_delta", "scarcity_delta"]
        },
        "error": result.get("error"),
    }
    rats = [out[k] for k in rat_cols if out[k] is not None]
    out["rationality_mean"] = round(sum(rats) / len(rats), 2) if rats else None
    revs = [out[k] or 0 for k in rev_cols]
    out["V_llm_revised"] = sum(revs)
    out["V_delta"] = out["V_llm_revised"] - (out["V_llm_raw"] or 0)
    return out


class Writer:
    """Incremental CSV writer: appends rows and flushes periodically."""

    def __init__(self, path: str, cols: list, every: int = 10):
        self.path = path
        self.cols = cols
        self.every = every
        self.buf: list[Dict] = []
        self.total = 0
        self.header = not os.path.exists(path)

    def add(self, row: Dict):
        self.buf.append(row)
        self.total += 1
        if len(self.buf) >= self.every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.cols)
            if self.header:
                w.writeheader()
                self.header = False
            w.writerows(self.buf)
        print(
            f"[save] flushed {len(self.buf)} rows -> {self.path} (total {self.total})"
        )
        self.buf.clear()


def load_done(path: str) -> set:
    """Read already-judged gene names from partial output (for resume)."""
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=["gene_name"])
    done = set(df["gene_name"].tolist())
    print(f"[resume] found {len(done)} already-judged genes in {path}")
    return done


async def run(
    df: pd.DataFrame, cell: str, concurrency: int, writer: Writer, client: AsyncOpenAI
):
    sem = asyncio.Semaphore(concurrency)

    async def task(row):
        async with sem:
            scores = {
                "hubness_score": row["hubness_score"],
                "phenotype_impact_score": row["phenotype_impact_score"],
                "knowledge_scarcity_score": row["knowledge_scarcity_score"],
            }
            print(f"Judging {row['gene_name']}...")
            result = await judge(row["gene_name"], cell, scores, client)
            writer.add(enrich(row, result))

    tasks = [asyncio.create_task(task(row)) for _, row in df.iterrows()]
    for coro in asyncio.as_completed(tasks):
        await coro
    writer.flush()


async def main():
    if not API_KEY:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Refusing to run with hardcoded credentials."
        )

    # --- health probe ---
    ok = await probe(BASE_URL, API_KEY)
    if not ok:
        print(f"[FATAL] API endpoint {BASE_URL} unreachable after retries. Abort.")
        return

    client = build_async_client()

    df_in = pd.read_csv(INPUT_FILE)
    valid = df_in[df_in["error"].isna()].copy()

    # --- resume support ---
    done = load_done(OUTPUT_FILE)
    todo = valid[~valid["gene_name"].isin(done)]
    print(f"Total: {len(valid)}, Already done: {len(done)}, Remaining: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        return

    writer = Writer(OUTPUT_FILE, COLUMNS, every=SAVE_EVERY)
    writer.header = len(done) == 0  # write header only on fresh start
    await run(todo, CELL_LINE, CONCURRENCY, writer, client)

    # --- summary ---
    df_out = pd.read_csv(OUTPUT_FILE)
    ok_n = df_out["error"].isna().sum()
    fail_n = df_out["error"].notna().sum()
    print(f"\n--- DONE ---")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Success: {ok_n}, Failed: {fail_n}")
    if "rationality_mean" in df_out.columns:
        print(f"Mean rationality: {df_out['rationality_mean'].mean():.2f}")
    if "V_delta" in df_out.columns:
        print(f"Mean |V_delta|: {df_out['V_delta'].abs().mean():.2f}")


if __name__ == "__main__":
    asyncio.run(main())

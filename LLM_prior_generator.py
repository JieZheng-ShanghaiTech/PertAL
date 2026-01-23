import openai
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict
from openai import OpenAI
from openai import APIError 



# =========================================================
# Note: Replace with your actual API Key 
# =========================================================
API_KEY = "Your own API"
PLATFORM_BASE_URL = "https://your-api-endpoint.com/v1" 

# 2. Experimental Context
MODEL_NAME = "gpt-4.1-mini"  
CELL_LINE = "K562"     
DATASET_NAME='K562'
# 3. Output and Input
OUTPUT_FILE = f"result/{DATASET_NAME}_{MODEL_NAME}_{CELL_LINE}_gene_priors.csv"
with open(f'{DATASET_NAME}_{CELL_LINE}_genes.txt') as f:
    CANDIDATE_GENES = [line.strip() for line in f]


# --- Initialize the Client ---
# This single line handles both Base_URL replacement and API KEY (Bearer Token) setup.
try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=f"https://{PLATFORM_BASE_URL}/v1" 
    )

except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your API_KEY and PLATFORM_BASE_URL are correctly set.")
    exit()


def generate_structured_prompt(gene_name: str, cell_context: str):
    """
    Generates the System and User prompts with clear constraints for high-fidelity extraction.
    """
    # System Prompt: Sets the persona and rules
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


def get_llm_score(gene_name: str, cell_context: str,client: OpenAI) -> Dict:
    """
    Calls the OpenAI API to get the structured score, including error handling.
    """
    system_prompt, user_prompt = generate_structured_prompt(gene_name, cell_context)
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            # Main API Call with JSON enforcement
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # Crucial for structured output:
                response_format={"type": "json_object"}, 
                temperature=0.1
            )
            
            # Parse JSON and validate keys
            content = response.choices[0].message.content
            result_json = json.loads(content)
            
            required_keys = ['hubness_score', 'phenotype_impact_score', 'knowledge_scarcity_score']
            if all(key in result_json for key in required_keys):
                result_json['gene_name'] = gene_name # Ensure gene name is present
                return result_json
            else:
                raise ValueError("JSON output is incomplete or missing required score keys.")

        
        except Exception as e:
            print(f"[{gene_name}] Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            print("Raw LLM output:", content) 
    return {}




def batch_query(genes: List[str], cell_context: str,client: OpenAI) -> pd.DataFrame:
    """
    Main execution loop for batch processing with rate limiting.
    """
    results = []
    
    for i, gene in enumerate(genes):
        print(f"Processing {i+1}/{len(genes)}: Querying {gene}...")
        score_data = get_llm_score(gene, cell_context, client)
        results.append(score_data)
        
        # Rate Limiting: Essential for batch API calls
        time.sleep(1.2) 

    return pd.DataFrame(results)




if __name__ == "__main__":
    
    print(f"Starting batch query for {len(CANDIDATE_GENES)} genes using {MODEL_NAME}...")
    
    df_results = batch_query(CANDIDATE_GENES, CELL_LINE,client)
    if "error" not in df_results.columns:
        df_results["error"] = None
    
 
   
    score_columns = ['hubness_score', 'phenotype_impact_score', 'knowledge_scarcity_score']
    
    # Fill NaN scores (from failed queries) with 0 for aggregation, if necessary
    df_results[score_columns] = df_results[score_columns].fillna(0)
    
    # Simple summation for the raw prior score V_llm
    df_results['V_llm_raw'] = df_results[score_columns].sum(axis=1)
    
    # Store results
    df_results.to_csv(OUTPUT_FILE, index=False)
    
    success_count = len(df_results) - df_results['error'].notna().sum()
    error_count = df_results['error'].notna().sum()
    
    print(f"\n--- TASK COMPLETE ---")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Successfully queried {success_count} genes.")
    print(f"Failed {error_count} genes. Please review the output file for error messages.")
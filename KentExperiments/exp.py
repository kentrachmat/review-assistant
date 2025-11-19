import os
import json
import argparse
from pathlib import Path
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument(
    "--base_dir",
    type=str,
    default="/home/brachmat/phd/khuong/data/abstract/flaw_v1_short",
    help="Path to the base directory containing the papers."
)
args = parser.parse_args()

endpoint = os.environ.get("OPENAI_API_BASE", "XXX")
api_key = os.environ.get("OPENAI_API_KEY", "XXX")
deployment_name = os.environ.get("OPENAI_DEPLOYMENT_NAME", "XXX")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key,
)

BASE_DIR = Path(args.base_dir)
REVIEWS_DIR = BASE_DIR / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)


# Rewrite the given abstract into an overclaimed, buzzword-heavy, exaggerated version that intentionally violates good scientific practice. The rewritten abstract should aggressively overstate contributions, use excessive hype language, falsely claim strong novelty, inflate the scope to suggest universal applicability, and deliberately misalign with what the original abstract actually delivers. Produce one scientific-sounding but clearly overblown abstract.

# <METRICS>
# [Evaluation focus: abstract]
# - Verify that the abstract accurately represent the paper's actual contributions.
# - Check whether the stated claims are supported by the results and do not overstate generality.
# - Ensure the contributions are clearly articulated, with key assumptions or limitations acknowledged.
# - Evaluate whether the scope described in the abstract matches what the paper actually delivers.
# - Assess the originality and significance of the claimed novelty relative to prior work.
# </METRICS>

# <PENALTY_RULES>
# - If the abstract fundamentally misrepresents the actual contributions, assign score = 1.
# - If the core claims are clearly unsupported or contradicted by the results, assign score = 1.
# - If the claimed novelty is false, trivial, or a near-duplicate of prior work without acknowledgment, assign score = 1.
# - If the stated scope in the intro/abstract is severely misaligned with what the paper actually delivers, assign score = 1.
# - For less severe issues (e.g., minor overclaiming, incomplete contributions, moderate vagueness), adjust the score downward proportionally.
# </PENALTY_RULES>

# <METRICS>
# [Evaluation focus: results and analysis]
# - Check whether experimental results are clearly presented, interpretable, and consistent across tables and figures.
# - Verify that tables, metrics, and comparisons are meaningful, correctly computed, and relevant to the claims.
# - Evaluate whether baselines, ablations, and experimental design adequately support the conclusions.
# - Assess whether theoretical claims align with empirical outcomes (no contradictions or unexplained gaps).
# - Determine whether the analysis provides genuine insight rather than superficial or selective reporting.
# </METRICS>

# <PENALTY_RULES>
# - If tables, metrics, or reported results are mathematically incorrect, internally inconsistent, or impossible, assign score = 1.
# - If theoretical claims or stated theorems are invalid, contradictory, or unsupported by their own proofs, assign score = 1.
# - If experimental results contradict the paper’s conclusions without explanation, assign score = 1.
# - If key baselines or comparisons are missing in a way that invalidates the conclusions, assign score = 1.
# - For moderate issues (e.g., incomplete analysis, unclear figures, weak baseline selection), adjust the score downward proportionally.
# </PENALTY_RULES>

PROMPT_TEMPLATE = """You are an expert in the topic of "{paper_title}", known for exceptionally thorough, incisive, and constructive critiques.

<METRICS>
[Evaluation focus: abstract]
- Verify that the abstract accurately represent the paper's actual contributions.
- Check whether the stated claims are supported by the results and do not overstate generality.
- Ensure the contributions are clearly articulated, with key assumptions or limitations acknowledged.
- Evaluate whether the scope described in the abstract matches what the paper actually delivers.
- Assess the originality and significance of the claimed novelty relative to prior work.
- For less severe issues (e.g., minor overclaiming, incomplete contributions, moderate vagueness), adjust the score downward proportionally.
</METRICS>

<INSTRUCTIONS>
1. Carefully read and internalize the entire paper.
3. Adopt the comprehensive, critical reviewer persona described above.
3. Pay particular attention to the METRICS above.
4. You must apply the PENALTY_RULES strictly.
5. Generate exactly ONE review, formatted as a single valid JSON object that fills ALL fields in the schema below, uses EXACTLY the keys specified in the schema (no extra keys), and includes NO markdown, NO backticks, and NO explanatory text before or after the JSON object.
</INSTRUCTIONS>

<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>

<OUTPUT_FORMAT>
Required JSON schema (STRICT). The JSON object must have EXACTLY these keys:

* "weaknesses": array of strings
  - Each element is a single, concise weakness (no nested lists, no numbering, no markdown).
  - Be specific and actionable (e.g., "Experimental setup lacks baselines X and Y", not just "Experiments are weak").
  - If you genuinely find no weaknesses, return an empty array [].

* "strengths": array of strings
  - Each element is a single, concise strength (no nested lists, no numbering, no markdown).
  - Focus on concrete aspects: novelty, clarity, rigor, empirical support, impact, etc.
  - If you genuinely find no strengths, return an empty array [].

* "score": integer
  - reflecting paper quality based on the metrics.
- Must be one of: 9, 8, 7, 6, 5, 4, 3, 2, 1
- Interpretation:
- 9 = Outstanding: exceptional quality, well above top-tier conference standards
- 8 = Excellent: strong contribution with only minor weaknesses
- 7 = Very Good: solid work; some limitations but still clearly valuable
- 6 = Good: acceptable with noteworthy issues; could be improved
- 5 = Fair: borderline; mixed strengths and weaknesses
- 4 = Weak: below expected standards; several significant issues
- 3 = Poor: major flaws in novelty, rigor, or clarity
- 2 = Very Poor: severe methodological or conceptual errors
- 1 = Fundamentally Unsound: fails to meet basic scientific criteria
</OUTPUT_FORMAT>

Remember:
- Output ONLY the JSON object, nothing else.
- Ensure the JSON is strictly valid (use double quotes for all keys and string values, no trailing commas).
"""


def review_all_control_papers():
    for subdir in sorted(BASE_DIR.iterdir()):
        if subdir == "reviews":
            continue
        if not subdir.is_dir():
            continue

        paper_md = subdir / "paper.md"
        if not paper_md.exists():
            print(f"[SKIP] No paper.md in {subdir}")
            continue

        print(f"[INFO] Reviewing paper in: {subdir}")

        with open(paper_md, "r", encoding="utf-8") as f:
            paper_content = f.read()
            paper_title = paper_content.split("\n")[0][1:].strip()
            print(paper_title)
        full_prompt = PROMPT_TEMPLATE.format(paper_content=paper_content, paper_title=paper_title)

        completion = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            temperature=0.3,
        )

        raw_reply = completion.choices[0].message.content.strip()

        try:
            review_json = json.loads(raw_reply)
        except json.JSONDecodeError:
            print(f"[WARN] JSON decode failed for {subdir.name}, saving raw text.")
            review_json = {"raw_response": raw_reply}

        out_path = REVIEWS_DIR / f"{subdir.name}_review.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(review_json, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved review to: {out_path}")

if __name__ == "__main__":
    review_all_control_papers()
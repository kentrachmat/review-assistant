import os
import json
from pathlib import Path

def main():
    input_dir = Path("/home/brachmat/phd/khuong/data/metrics/flaw_v1/reviews_prompt_metrics")
    output_dir = Path("/home/brachmat/phd/khuong/data/metrics/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = {
        "all_weaknesses": [],
        "all_strengths": []
    }

    # Iterate through json files
    for file in input_dir.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Append weaknesses/strengths if exist
            if "weaknesses" in data and isinstance(data["weaknesses"], list):
                aggregated["all_weaknesses"].extend(data["weaknesses"])

            if "strengths" in data and isinstance(data["strengths"], list):
                aggregated["all_strengths"].extend(data["strengths"])

        except Exception as e:
            print(f"⚠️ Could not process {file.name}: {e}")

    # Save results
    output_path = output_dir / "aggregated_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"✅ Aggregated file saved to: {output_path}")


if __name__ == "__main__":
    main()

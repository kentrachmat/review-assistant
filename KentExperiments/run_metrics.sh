#!/bin/bash

python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/metrics/flaw_v1"

CONTROL_DIR="/home/brachmat/phd/khuong/data/control/reviews"
SHAM_DIR="/home/brachmat/phd/khuong/data/metrics/sham/reviews"

BASE_FLAW="/home/brachmat/phd/khuong/data/metrics"
OUTPUT_BASE="/home/brachmat/phd/khuong/data/metrics/evaluation_results"

for version in flaw_v1; do
    FLAW_DIR="$BASE_FLAW/$version/reviews_prompt_abstract"
    
    TITLE="Metrics Manipulations – ${version}"
    
    # Output dir
    OUT_DIR="${OUTPUT_BASE}/${version}"
    mkdir -p "$OUT_DIR"

    echo "====================================================="
    echo "Running evaluation for $version"
    echo "Flaw dir: $FLAW_DIR"
    echo "Output: $OUT_DIR"
    echo "====================================================="

    python eval.py \
        --control_dir "$CONTROL_DIR" \
        --flaw_dir "$FLAW_DIR" \
        --sham_dir "$SHAM_DIR" \
        --output_dir "$OUT_DIR" \
        --plot_title "$TITLE" \
        --output_prefix "$version"
done



# CONTROL_DIR="/home/brachmat/phd/khuong/data/control/reviews"
# SHAM_DIR="/home/brachmat/phd/khuong/data/metrics/sham/reviews"

# BASE_FLAW="/home/brachmat/phd/khuong/data/metrics"
# OUTPUT_BASE="/home/brachmat/phd/khuong/data/metrics/evaluation_results_distribution"

# for version in flaw_v1; do
#     FLAW_DIR="$BASE_FLAW/$version/reviews_prompt_abstract"
    
#     TITLE="Metrics Manipulations – ${version}"

#     # Output dir
#     OUT_DIR="${OUTPUT_BASE}/${version}"
#     mkdir -p "$OUT_DIR"

#     echo "====================================================="
#     echo "Running evaluation for $version"
#     echo "Flaw dir: $FLAW_DIR"
#     echo "Output: $OUT_DIR"
#     echo "====================================================="

#     python eval_with_distribution.py \
#         --control_dir "$CONTROL_DIR" \
#         --flaw_dir "$FLAW_DIR" \
#         --sham_dir "$SHAM_DIR" \
#         --output_dir "$OUT_DIR" \
#         --plot_title "$TITLE" \
#         --output_prefix "$version"
# done
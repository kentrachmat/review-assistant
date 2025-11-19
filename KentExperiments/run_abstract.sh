#!/bin/bash

python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/control"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/flaw_v1_short"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/flaw_v1"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/flaw_v2"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/flaw_v3"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/flaw_v4"
python3 exp.py --base_dir "/home/brachmat/phd/khuong/data/abstract/sham"


CONTROL_DIR="/home/brachmat/phd/khuong/data/control/reviews"
SHAM_DIR="/home/brachmat/phd/khuong/data/abstract/sham/reviews"

BASE_FLAW="/home/brachmat/phd/khuong/data/abstract"
OUTPUT_BASE="/home/brachmat/phd/khuong/data/abstract/evaluation_results"

for version in flaw_v1_short flaw_v1 flaw_v2 flaw_v3 flaw_v4 ; do
    FLAW_DIR="$BASE_FLAW/$version/reviews"
    
    TITLE="Abstract Manipulations – ${version}"
    
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
# SHAM_DIR="/home/brachmat/phd/khuong/data/abstract/sham/reviews"

# BASE_FLAW="/home/brachmat/phd/khuong/data/abstract"
# OUTPUT_BASE="/home/brachmat/phd/khuong/data/abstract/evaluation_results_distribution"

# for version in flaw_v1_short flaw_v1 flaw_v2 flaw_v3 flaw_v4 ; do
#     FLAW_DIR="$BASE_FLAW/$version/reviews"
    
#     TITLE="Abstract Manipulations – ${version}"
    
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
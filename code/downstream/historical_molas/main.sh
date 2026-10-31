#!/bin/bash
set -euo pipefail

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TORCH_NCCL_ENABLE_MONITORING=0

PYTHON="/srv/conda/envs/as_bbo/as_dock/bin/python"

#----------------------------------------Model and Benchmark Settings----------------------------------------#
# Devices used by DDP training. Testing uses the first device only.
devices=(0 1 2 5 6 7)

# Benchmark to train/test
benchmark="moad"

# Relaxation setting for the benchmark
relaxation='both'  # 'both', 'true', or 'false'

# Model to train
model='MolAS'

# Label sources
train_label_source='al'         # usually 'composite' or 'al'
test_label_source='composite'   # default: report against ground-truth-derived labels

# Fold-specific AL matrices and the matching exported MolAS split.
al_matrices_root="/data1/home/jw1017/AS_Dock/data/MolAS/AL/PORT-AS_cost_aware"
al_seed=0
split_export_path="/data1/home/jw1017/AS_Dock/MolAS_AL/split_exports/moad_5fold_split_seed42.csv"
verify_fold_inputs='true'

# Methods/pcts to run when train_label_source='al'.
# al_method_list=(
#   # 'cheapest_first_svd'
#   # 'cost_aware_ensemble_disagreement_svd'
#   # 'cost_aware_expected_regret_svd'
#   # 'cost_aware_margin_active_svd'
#   # 'cost_aware_qin2025_mean_ucb_svd'
#   # 'cost_aware_qin2025_mpe_svd'
#   # 'cost_aware_qin2025_uncertainty_svd'
#   'margin_active_svd'
#   'ensemble_disagreement_svd'
#   'expected_regret_svd'
#   'qin2025_mean_ucb_svd'
#   'qin2025_mpe_svd'
#   # 'qin2025_uncertainty_svd'
# )
al_method_list=(
  # 'plain_mean_ucb'
  # 'direct_cost_uc'
  # 'relaxed_cost_power_05'
  # 'log_cost'
  'runtime_uncertainty'
  'cost_tiers_5'
  # 'hybrid_b50'
)
# pct_list=(
#   # 01
#   # 02
#   # 05
#   # 10
#   12
#   14
#   # 15
#   16
#   18
#   21
#   22
#   23
#   24
#   # 25
#   26
#   27
#   28
#   29
#   # 30
#   # 35
#   # 40
#   # 45
#   # 50
#   55
#   # 60
#   65
#   # 70
#   75
#   # 80
#   85
#   # 90
#   95
#   # 100
# )
pct_list=(
  70
  80
  90
  100
)
matrix_run_specs=()

seed=42
max_epochs=300
batch_size=64
num_workers_train=0
num_workers_test=1
learning_rate=1e-3
skip_existing_train='true'  # 'true' to skip fold training when completion marker or best ckpt exists
train_done_marker='.train_done'

log_root="/data1/home/jw1017/AS_Dock/results/MolAS/lightning_logs"
test_output_root="/data1/home/jw1017/AS_Dock/results/MolAS/test_outputs"
#------------------------------------------------------------------------------------------------------------#

sanitize_component() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' /' '__' | tr -s '_'
}

format_pct_token() {
  printf "%03d" "$((10#$1))"
}

build_matrix_run_specs() {
  matrix_run_specs=()
  for method in "${al_method_list[@]}"; do
    if [[ -z "$method" ]]; then
      echo "Error: al_method_list contains an empty method."
      exit 1
    fi

    for pct in "${pct_list[@]}"; do
      if [[ ! "$pct" =~ ^[0-9]+$ ]]; then
        echo "Error: pct_list contains non-integer value: '$pct'"
        exit 1
      fi

      local pct_token
      pct_token=$(format_pct_token "$pct")
      matrix_run_specs+=("${method}|${pct_token}")
    done
  done
}

fold_matrix_path() {
  local method="$1"
  local fold="$2"
  local pct_token="$3"
  echo "${al_matrices_root}/${method}/fold${fold}/seed${al_seed}/strict_score_matrices/strict_score_${pct_token}pct.csv"
}

verify_al_run() {
  local method="$1"
  local pct_token="$2"

  "$PYTHON" - "$split_export_path" "$al_matrices_root" "$method" "$al_seed" "$pct_token" "$seed" <<'PY'
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

split_path, root, method, al_seed, pct, molas_seed = sys.argv[1:]
split_df = pd.read_csv(split_path)

required = {"instance_id", "test_fold"}
missing_columns = required.difference(split_df.columns)
if missing_columns:
    raise SystemExit(f"Split export is missing columns: {sorted(missing_columns)}")
if split_df["instance_id"].duplicated().any():
    raise SystemExit("Split export contains duplicate instance_id values.")

exported_folds = split_df["test_fold"].astype(int).tolist()
generated_folds = [-1] * len(split_df)
for fold, (_, test_idx) in enumerate(
    KFold(n_splits=5, shuffle=True, random_state=int(molas_seed)).split(split_df)
):
    for idx in test_idx:
        generated_folds[idx] = fold
if generated_folds != exported_folds:
    raise SystemExit(
        f"Split export does not match MolAS KFold(n_splits=5, shuffle=True, "
        f"random_state={molas_seed})."
    )

for fold in range(5):
    matrix_path = (
        Path(root) / method / f"fold{fold}" / f"seed{al_seed}"
        / "strict_score_matrices" / f"strict_score_{pct}pct.csv"
    )
    if not matrix_path.is_file():
        raise SystemExit(f"Missing fold matrix: {matrix_path}")

    matrix_df = pd.read_csv(matrix_path, usecols=["PDB_CCD_ID"])
    actual = matrix_df["PDB_CCD_ID"].astype(str).tolist()
    expected = split_df.loc[split_df["test_fold"].ne(fold), "instance_id"].astype(str).tolist()

    if len(actual) != len(set(actual)):
        raise SystemExit(f"Fold {fold} matrix contains duplicate PDB_CCD_ID values.")
    if set(actual) != set(expected):
        missing = sorted(set(expected).difference(actual))[:5]
        unexpected = sorted(set(actual).difference(expected))[:5]
        raise SystemExit(
            f"Fold {fold} row-set mismatch: missing={missing}, unexpected={unexpected}"
        )
    if actual != expected:
        raise SystemExit(
            f"Fold {fold} has the correct rows but not the exported training-row order."
        )

print(f"Verified {method} at {pct}%: all five matrices match the exported training folds.")
PY
}

has_best_checkpoint() {
  local ckpt_dir="$1"
  local best_ckpt
  best_ckpt=$(find "$ckpt_dir" -type f -name 'best-epoch=*-*.ckpt' 2>/dev/null | head -n1 || true)
  [[ -n "$best_ckpt" ]]
}

num_devices=${#devices[@]}

declare -A num_classes_both_dict=(
  [posex]=48
  [astex]=48
  [astex_posex]=48
  [moad]=7
  [posex_self_docking]=48
  [posex_cross_docking]=48
  [posebusters]=14
  [posebench]=14
)
declare -A num_classes_dict=(
  [posex]=24
  [astex]=24
  [astex_posex]=24
  [moad]=7
  [posex_self_docking]=24
  [posex_cross_docking]=24
  [posebusters]=7
  [posebench]=7
)
declare -A incl_columns_dict=(
  [posex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [astex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [astex_posex]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [moad]=''
  [posex_self_docking]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [posex_cross_docking]='surfdock unimol diffdock diffdock_l gnina autodock glide alphafold3 moe dynamicbind rfaa fabind equibind boltz1x tankbind neuralplexer chai deepdock boltz ifd protenix diffdock_pocket DS interformer'
  [posebusters]='deepdock diffdock equibind gold tankbind unimol autodock'
)

if [[ "$relaxation" == "both" ]]; then
  incl_columns=''
  num_classes=${num_classes_both_dict[$benchmark]}
elif [[ "$relaxation" == "true" ]]; then
  incl_columns=""
  for method in ${incl_columns_dict[$benchmark]}; do
    incl_columns+="${method}_relaxation "
  done
  num_classes=${num_classes_dict[$benchmark]}
else
  incl_columns="${incl_columns_dict[$benchmark]}"
  num_classes=${num_classes_dict[$benchmark]}
fi
read -r -a incl_column_args <<< "$incl_columns"

echo "Number of classes for benchmark $benchmark: $num_classes"

if [[ "$model" == "MCGNNASDock" ]]; then
  num_node_features=960
else
  num_node_features=1152
fi

if [[ "$train_label_source" == "al" ]]; then
  if [[ ${#al_method_list[@]} -eq 0 || ${#pct_list[@]} -eq 0 ]]; then
    echo "Error: when train_label_source='al', al_method_list and pct_list must both be non-empty."
    exit 1
  fi
  build_matrix_run_specs
else
  matrix_run_specs=("|")
  echo "No score matrices provided. Running with default training labels only."
fi

matrix_run_index=0
for run_spec in "${matrix_run_specs[@]}"; do
  matrix_run_index=$((matrix_run_index + 1))
  echo "=== Matrix run ${matrix_run_index}/${#matrix_run_specs[@]} ==="

  train_matrix_tag="default"

  if [[ "$train_label_source" == "al" ]]; then
    IFS='|' read -r train_method train_pct_token <<< "$run_spec"
    if [[ "$verify_fold_inputs" == "true" ]]; then
      verify_al_run "$train_method" "$train_pct_token"
    fi
    train_matrix_tag="port_as/$(sanitize_component "$train_method")/seed${al_seed}/b${train_pct_token}_pct"
  fi

  log_name="benchmark_experiment_${benchmark}/train_${train_label_source}/${train_matrix_tag}"
  test_output_dir="${test_output_root}/${log_name}/test_${test_label_source}/default"

  echo "Resolved log_name: $log_name"
  mkdir -p "$test_output_dir"

  echo "--- Training (folds 0-4) ---"
  for i in {0..4}; do
    base_log_dir="${log_root}/${log_name}"
    fold_log_dir="${base_log_dir}/${i}"
    marker_path="${fold_log_dir}/${train_done_marker}"

    if [[ "$skip_existing_train" == "true" ]]; then
      if [[ -f "$marker_path" ]]; then
        if has_best_checkpoint "${fold_log_dir}/checkpoints"; then
          echo "Skipping training fold $i: marker + checkpoint found"
          continue
        fi
        echo "Found stale marker for fold $i (no checkpoint). Removing marker and retraining."
        rm -f "$marker_path"
      fi
      if has_best_checkpoint "${fold_log_dir}/checkpoints"; then
        echo "Skipping training fold $i: found existing best checkpoint in ${fold_log_dir}/checkpoints"
        continue
      fi
    fi

    echo "Training with fold $i"

    current_score_matrix_path=""
    train_split_args=(--k_folds 5 --fold_num "$i" --test_size 0.2)
    train_score_args=()
    if [[ "$train_label_source" == "al" ]]; then
      current_score_matrix_path=$(fold_matrix_path "$train_method" "$i" "$train_pct_token")
      train_split_args=(--fold_num "$i" --test_size 0)
      train_score_args=(--score_matrix_path "$current_score_matrix_path")
      echo "Fold-specific score_matrix_path: $current_score_matrix_path"
    fi

    train_cmd=(
      "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$num_devices" main.py
      "${train_split_args[@]}"
      --num_node_features "$num_node_features"
      --max_epochs "$max_epochs"
      --devices "${devices[@]}"
      --num_classes "$num_classes"
      --incl_columns "${incl_column_args[@]}"
      --batch_size "$batch_size"
      --num_workers "$num_workers_train"
      --lr "$learning_rate"
      --benchmark "$benchmark"
      --label_source "$train_label_source"
      --seed "$seed"
      --log_root "$log_root"
      --log_name "$log_name"
      --log_version "$i"
      --model "$model"
      "${train_score_args[@]}"
    )

    max_launch_attempts=3

    for attempt in $(seq 1 "$max_launch_attempts"); do
      echo "DDP launch attempt ${attempt}/${max_launch_attempts}"

      if LOGLEVEL=INFO "${train_cmd[@]}"; then
        break
      fi

      if [[ "$attempt" -eq "$max_launch_attempts" ]]; then
        echo "Error: training failed after ${max_launch_attempts} attempts."
        exit 1
      fi

      echo "DDP launch failed; waiting 5 seconds and retrying..."
      sleep 5
    done

    mkdir -p "$(dirname "$marker_path")"
    {
      echo "done_at=$(date -Is)"
      echo "benchmark=${benchmark}"
      echo "fold=${i}"
      echo "train_label_source=${train_label_source}"
      echo "score_matrix_path=${current_score_matrix_path}"
      echo "log_name=${log_name}"
    } > "$marker_path"
  done

  echo "--- Testing (folds 0-4) ---"
  after_ckpts=()

  for i in {0..4}; do
    base_log_dir="${log_root}/${log_name}"
    fold_log_dir="${base_log_dir}/${i}"
    dir="${fold_log_dir}/checkpoints"
    fold_output_dir="${test_output_dir}/fold_${i}"
    mkdir -p "$fold_output_dir"

    best_ckpt=$(find "$dir" -type f -name 'best-epoch=*-*.ckpt' | \
      grep -oE 'best-epoch=([0-9]+)-[^/]*\.ckpt$' | \
      sed -E 's/^best-epoch=([0-9]+)-.*/\1 &/' | \
      sort -nr | \
      head -n1 | \
      cut -d' ' -f2- || true)

    if [[ -z "$best_ckpt" ]]; then
      echo "Error: no best checkpoint found for fold $i in $dir"
      exit 1
    fi

    best_ckpt_path="$dir/$best_ckpt"
    after_ckpts+=("$best_ckpt_path")
    echo "#---------------------------Testing with checkpoint: $best_ckpt_path---------------------------#"

    "$PYTHON" main.py \
      --k_folds 5 \
      --fold_num "$i" \
      --num_node_features "$num_node_features" \
      --devices "${devices[0]}" \
      --num_classes "$num_classes" \
      --incl_columns "${incl_column_args[@]}" \
      --num_workers "$num_workers_test" \
      --benchmark "$benchmark" \
      --label_source "$test_label_source" \
      --seed "$seed" \
      --test_size 0.2 \
      --test \
      --ckpt_path "$best_ckpt_path" \
      --model "$model"

    ckpt_prefix="${best_ckpt_path%.ckpt}"
    for ext in \
      "_predicted_scores.csv" \
      ".csv" \
      "_full.csv" \
      "_descriptive.csv" \
      "_algorithm_selection_counts.csv"; do
      src="${ckpt_prefix}${ext}"
      if [[ -f "$src" ]]; then
        cp -f "$src" "$fold_output_dir/"
      fi
    done
  done

  echo "--- Overall summary ---"
  summary_csv="${test_output_dir}/overall_summary_metrics.csv"
  algorithm_counts_csv="${test_output_dir}/overall_algorithm_selection_counts.csv"
  overall_predicted_scores_csv="${test_output_dir}/overall_predicted_scores.csv"

  "$PYTHON" stats_test.py \
    --after_ckpts "${after_ckpts[@]}" \
    --summary_csv "$summary_csv" \
    --algorithm_counts_csv "$algorithm_counts_csv"

  "$PYTHON" - <<'PY' "$test_output_dir" "$overall_predicted_scores_csv"
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
frames = []
for fold_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('fold_')]):
    files = sorted(fold_dir.glob('*_predicted_scores.csv'))
    for file in files:
        df = pd.read_csv(file)
        df.insert(0, 'fold', fold_dir.name)
        df.insert(1, 'source_file', file.name)
        frames.append(df)

if frames:
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
PY

  echo "Completed matrix run ${matrix_run_index}. Outputs saved to: $test_output_dir"
done

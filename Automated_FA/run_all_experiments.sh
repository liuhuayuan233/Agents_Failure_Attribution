#!/bin/bash
set -e
cd "$(dirname "$0")"

MODEL="gpt-5.4-nano"
AG_DIR="../Who_and_When/Algorithm-Generated"
HC_DIR="../Who_and_When/Hand-Crafted"
mkdir -p results outputs

echo "=========================================="
echo "  开始全部实验 (里程碑1 + 里程碑2)"
echo "  模型: $MODEL"
echo "  时间: $(date)"
echo "=========================================="

# --- Feedback 方法 (E0, E1, E2) ---

echo -e "\n>>> [1/18] E0 AG (generic, bypass_intent)"
python benchmark_evaluator.py --data_dir "$AG_DIR" --is_handcrafted False \
  --feedback_level generic --bypass_intent True --prompt_variant benchmark \
  --output results/E0_ag.json

echo -e "\n>>> [2/18] E0 HC"
python benchmark_evaluator.py --data_dir "$HC_DIR" --is_handcrafted True \
  --feedback_level generic --bypass_intent True --prompt_variant benchmark \
  --output results/E0_hc.json

echo -e "\n>>> [3/18] E1 AG (task_aware, bypass_intent)"
python benchmark_evaluator.py --data_dir "$AG_DIR" --is_handcrafted False \
  --feedback_level task_aware --bypass_intent True --prompt_variant benchmark \
  --output results/E1_ag.json

echo -e "\n>>> [4/18] E1 HC"
python benchmark_evaluator.py --data_dir "$HC_DIR" --is_handcrafted True \
  --feedback_level task_aware --bypass_intent True --prompt_variant benchmark \
  --output results/E1_hc.json

echo -e "\n>>> [5/18] E2 AG (task_aware, full intent)"
python benchmark_evaluator.py --data_dir "$AG_DIR" --is_handcrafted False \
  --feedback_level task_aware --bypass_intent False --prompt_variant benchmark \
  --output results/E2_ag.json

echo -e "\n>>> [6/18] E2 HC"
python benchmark_evaluator.py --data_dir "$HC_DIR" --is_handcrafted True \
  --feedback_level task_aware --bypass_intent False --prompt_variant benchmark \
  --output results/E2_hc.json

# --- 原论文方法 (B1-B6) ---
# 先用 inference.py 生成 txt, 再用 eval_baseline 评测

run_baseline() {
  local tag=$1 method=$2 gt_flag=$3 dir=$4 hc=$5 suffix=$6 num=$7
  echo -e "\n>>> [$num/18] $tag"
  python inference.py --method "$method" --model "$MODEL" \
    --directory_path "$dir" --is_handcrafted "$hc" $gt_flag
  local txt="outputs/${method}_${MODEL}_${suffix}.txt"
  python benchmark_evaluator.py --eval_baseline "$txt" \
    --data_dir "$dir" --is_handcrafted "$hc" \
    --method_name "$tag" --output "results/${tag}.json"
}

run_baseline "B1_ag" all_at_once "" "$AG_DIR" False alg_generated "7"
run_baseline "B1_hc" all_at_once "" "$HC_DIR" True handcrafted "8"
run_baseline "B2_ag" step_by_step "" "$AG_DIR" False alg_generated "9"
run_baseline "B2_hc" step_by_step "" "$HC_DIR" True handcrafted "10"
run_baseline "B3_ag" binary_search "" "$AG_DIR" False alg_generated "11"
run_baseline "B3_hc" binary_search "" "$HC_DIR" True handcrafted "12"
run_baseline "B4_ag" all_at_once "--no_ground_truth" "$AG_DIR" False alg_generated "13"
run_baseline "B4_hc" all_at_once "--no_ground_truth" "$HC_DIR" True handcrafted "14"
run_baseline "B5_ag" step_by_step "--no_ground_truth" "$AG_DIR" False alg_generated "15"
run_baseline "B5_hc" step_by_step "--no_ground_truth" "$HC_DIR" True handcrafted "16"
run_baseline "B6_ag" binary_search "--no_ground_truth" "$AG_DIR" False alg_generated "17"
run_baseline "B6_hc" binary_search "--no_ground_truth" "$HC_DIR" True handcrafted "18"

# --- 可视化 ---
echo -e "\n>>> 生成可视化图表"
python benchmark_evaluator.py --visualize \
  --result_files results/E0_ag.json results/E1_ag.json results/E2_ag.json \
    results/B1_ag.json results/B2_ag.json results/B3_ag.json \
    results/B4_ag.json results/B5_ag.json results/B6_ag.json \
  --output_dir results/figures/

echo -e "\n=========================================="
echo "  全部实验完成! $(date)"
echo "=========================================="

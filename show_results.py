"""Show evaluation results for all run instances.

Usage:
  python show_results.py [--traces-dir traces]
"""

import os
import re
import csv
import json
import glob
import argparse
import math
import pandas as pd


def parse_instance_id(dir_name):
    m = re.match(r"^(?P<id>.+)_(\d{8})_(\d{6})$", dir_name)
    return m.group("id") if m else None


def parse_timestamp(dir_name):
    m = re.match(r"^(.+)_(\d{8})_(\d{6})$", dir_name)
    return f"{m.group(2)}_{m.group(3)}" if m else None


def has_sub_agents(trace_dir):
    sub_dir = os.path.join(trace_dir, "sub_agents")
    if not os.path.isdir(sub_dir):
        return False
    return any(f.endswith("_messages.json") for f in os.listdir(sub_dir))


def load_eval_standard(gold_dir):
    eval_jsonl = os.path.join(gold_dir, "spider2lite_eval.jsonl")
    if not os.path.exists(eval_jsonl):
        return {}
    data = {}
    with open(eval_jsonl) as f:
        for line in f:
            obj = json.loads(line.strip())
            data[obj["instance_id"]] = obj
    return data


def compare_pandas_table(pred, gold, condition_cols=[], ignore_order=False):
    tolerance = 1e-2

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
        if len(v1) != len(v2):
            return False
        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold
    pred_cols = pred

    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    for gold_vec in t_gold_list:
        if not any(vectors_match(gold_vec, pred_vec, ignore_order_=ignore_order) for pred_vec in t_pred_list):
            return False
    return True


def evaluate_instance(pred_csv, gold_dir, instance_id, eval_standard):
    gold_result_dir = os.path.join(gold_dir, "exec_result")
    pattern = re.compile(rf'^{re.escape(instance_id)}(_[a-z])?\.csv$')
    gold_files = sorted(f for f in os.listdir(gold_result_dir) if pattern.match(f))

    if not gold_files:
        return None  # no ground truth result

    if not os.path.exists(pred_csv):
        return 0

    try:
        pred_pd = pd.read_csv(pred_csv)
    except Exception:
        return 0

    eval_info = eval_standard.get(instance_id, {})
    condition_cols = eval_info.get("condition_cols", [])
    ignore_order = eval_info.get("ignore_order", False)

    gold_pds = [pd.read_csv(os.path.join(gold_result_dir, f)) for f in gold_files]

    for gold_pd in gold_pds:
        if compare_pandas_table(pred_pd, gold_pd, condition_cols, ignore_order):
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces-dir", default="traces")
    parser.add_argument("--gold-dir", default="evaluation/gold")
    args = parser.parse_args()

    eval_standard = load_eval_standard(args.gold_dir)
    gold_sql_dir = os.path.join(args.gold_dir, "sql")
    gold_result_dir = os.path.join(args.gold_dir, "exec_result")

    # Collect all traces per instance, separated by vanilla vs NL-UDF
    # Keep latest of each type per instance
    instance_vanilla = {}   # inst_id -> {"ts", "dir"}
    instance_nl = {}        # inst_id -> {"ts", "dir"}

    for entry in os.listdir(args.traces_dir):
        entry_path = os.path.join(args.traces_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        inst_id = parse_instance_id(entry)
        if not inst_id or not inst_id.startswith("local"):
            continue
        ts = parse_timestamp(entry)
        used_nl = has_sub_agents(entry_path)

        target = instance_nl if used_nl else instance_vanilla
        if inst_id not in target or ts > target[inst_id]["ts"]:
            target[inst_id] = {"ts": ts, "dir": entry_path}

    # All instance IDs
    all_ids = sorted(set(list(instance_vanilla.keys()) + list(instance_nl.keys())))

    def get_pred_csv(trace_info):
        if trace_info is None:
            return None
        path = os.path.join(trace_info["dir"], "execution_result.csv")
        return path if os.path.exists(path) else None

    def eval_trace(trace_info, inst_id):
        """Returns (ran: bool, score: int|None)"""
        if trace_info is None:
            return False, None
        pred_csv = get_pred_csv(trace_info)
        if not pred_csv:
            return True, None  # ran but no result (error)

        gt_result_pattern = re.compile(rf'^{re.escape(inst_id)}(_[a-z])?\.csv$')
        has_gt_result = any(gt_result_pattern.match(f) for f in os.listdir(gold_result_dir))
        if not has_gt_result:
            return True, None  # ran but no GT to compare

        return True, evaluate_instance(pred_csv, args.gold_dir, inst_id, eval_standard)

    def compare_vanilla_nl(v_info, nl_info):
        """Compare vanilla and NL-UDF results against each other. Returns 'same', 'diff', or '-'."""
        v_csv = get_pred_csv(v_info)
        nl_csv = get_pred_csv(nl_info)
        if not v_csv or not nl_csv:
            return "-"
        try:
            v_pd = pd.read_csv(v_csv)
            nl_pd = pd.read_csv(nl_csv)
            if compare_pandas_table(nl_pd, v_pd) and compare_pandas_table(v_pd, nl_pd):
                return "same"
            return "diff"
        except Exception:
            return "-"

    # Print table
    header = f"{'Instance':<12s} {'GT SQL':>7s} {'Vanilla':>8s} {'NL-UDF':>8s} {'V=NL':>6s}"
    print(header)
    print("-" * len(header))

    v_total = 0; v_correct = 0
    nl_total = 0; nl_correct = 0
    same_count = 0; diff_count = 0; both_ran = 0

    # Sort: instances with GT SQL first, then without
    all_ids.sort(key=lambda x: (not os.path.exists(os.path.join(gold_sql_dir, f"{x}.sql")), x))

    for inst_id in all_ids:
        has_gt_sql = os.path.exists(os.path.join(gold_sql_dir, f"{inst_id}.sql"))

        v_ran_flag, v_score = eval_trace(instance_vanilla.get(inst_id), inst_id)
        nl_ran_flag, nl_score = eval_trace(instance_nl.get(inst_id), inst_id)
        v_nl_match = compare_vanilla_nl(instance_vanilla.get(inst_id), instance_nl.get(inst_id))

        def fmt_score(ran, score):
            if not ran:
                return "-"
            if score is None:
                return "ERROR"
            return "PASS" if score == 1 else "FAIL"

        gt_sql_str = "yes" if has_gt_sql else "no"

        print(f"{inst_id:<12s} {gt_sql_str:>7s} {fmt_score(v_ran_flag, v_score):>8s} {fmt_score(nl_ran_flag, nl_score):>8s} {v_nl_match:>6s}")

        if v_ran_flag and v_score is not None:
            v_total += 1
            if v_score == 1:
                v_correct += 1
        if nl_ran_flag and nl_score is not None:
            nl_total += 1
            if nl_score == 1:
                nl_correct += 1
        if v_nl_match in ("same", "diff"):
            both_ran += 1
            if v_nl_match == "same":
                same_count += 1
            else:
                diff_count += 1

    print("-" * len(header))
    v_pct = f"{v_correct/v_total*100:.1f}%" if v_total else "N/A"
    nl_pct = f"{nl_correct/nl_total*100:.1f}%" if nl_total else "N/A"
    print(f"Vanilla:  {v_correct}/{v_total} ({v_pct})")
    print(f"NL-UDF:   {nl_correct}/{nl_total} ({nl_pct})")
    if both_ran:
        print(f"V vs NL:  {same_count} same, {diff_count} diff (out of {both_ran} both ran)")

    # Show cases where one method succeeded and the other didn't
    nl_wins = []
    v_wins = []
    for inst_id in all_ids:
        v_ran_flag, v_score = eval_trace(instance_vanilla.get(inst_id), inst_id)
        nl_ran_flag, nl_score = eval_trace(instance_nl.get(inst_id), inst_id)
        if nl_ran_flag and nl_score == 1 and (not v_ran_flag or v_score != 1):
            v_status = fmt_score(v_ran_flag, v_score)
            nl_wins.append((inst_id, v_status))
        if v_ran_flag and v_score == 1 and (not nl_ran_flag or nl_score != 1):
            nl_status = fmt_score(nl_ran_flag, nl_score)
            v_wins.append((inst_id, nl_status))

    if nl_wins:
        print(f"\nNL-UDF wins ({len(nl_wins)}) — NL-UDF PASS, Vanilla didn't:")
        for inst, v_st in nl_wins:
            print(f"  {inst:<12s}  Vanilla={v_st}")
    if v_wins:
        print(f"\nVanilla wins ({len(v_wins)}) — Vanilla PASS, NL-UDF didn't:")
        for inst, nl_st in v_wins:
            print(f"  {inst:<12s}  NL-UDF={nl_st}")


if __name__ == "__main__":
    main()

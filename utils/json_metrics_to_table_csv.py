# utils/json_metrics_to_table_csv.py
import argparse
import json
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="metrics.json produced by eval script")
    ap.add_argument("--out_csv", required=True, help="output csv (dataset-wise table)")
    ap.add_argument("--ndigits", type=int, default=3, help="rounding decimals")
    args = ap.parse_args()

    with open(args.in_json, "r") as f:
        metrics = json.load(f)

    rows = []
    for k, v in metrics.items():
        # skip overall blocks here, handle separately at end
        if k.startswith("__"):
            continue
        if not isinstance(v, dict):
            continue

        rows.append({
            "Dataset": k,
            "N": int(v.get("N", 0)),
            "AUC": float(v.get("AUC", 0.5)),
            "EER": float(v.get("EER", 0.5)),
            "BEST_ACC": float(v.get("BEST_ACC", 0.0)),
            "ACC_AT_EER": float(v.get("ACC_AT_EER", 0.0)),
            "EER_THR": float(v.get("EER_THR", 0.5)),
        })

    df = pd.DataFrame(rows)

    # sort datasets by name (or by N if you want)
    df = df.sort_values(by="Dataset").reset_index(drop=True)

    # add micro/macro rows like "Average" column in papers
    if "__overall_micro__" in metrics:
        om = metrics["__overall_micro__"]
        df = pd.concat([df, pd.DataFrame([{
            "Dataset": "__overall_micro__",
            "N": int(om.get("N", 0)),
            "AUC": float(om.get("AUC", 0.5)),
            "EER": float(om.get("EER", 0.5)),
            "BEST_ACC": float(om.get("BEST_ACC", 0.0)),
            "ACC_AT_EER": float(om.get("ACC_AT_EER", 0.0)),
            "EER_THR": float(om.get("EER_THR", 0.5)),
        }])], ignore_index=True)

    if "__overall_macro__" in metrics:
        oc = metrics["__overall_macro__"]
        # macro only has AUC/EER (by your definition)
        df = pd.concat([df, pd.DataFrame([{
            "Dataset": "__overall_macro__",
            "N": "",
            "AUC": float(oc.get("AUC_macro", 0.5)),
            "EER": float(oc.get("EER_macro", 0.5)),
            "BEST_ACC": "",
            "ACC_AT_EER": "",
            "EER_THR": "",
        }])], ignore_index=True)

    # rounding (3 decimals like your example)
    for col in ["AUC", "EER", "BEST_ACC", "ACC_AT_EER", "EER_THR"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(args.ndigits)

    df.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    print(df)

if __name__ == "__main__":
    main()
    
    
    
# python utils/json_metrics_to_table_csv.py \
#   --in_json experiments/FFPP_10/eval_all/generalization_all_test_metrics.json \
#   --out_csv experiments/FFPP_10/eval_all/generalization_table.csv \
#   --ndigits 3
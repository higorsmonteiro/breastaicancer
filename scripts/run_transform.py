import argparse
import subprocess
from typing import List

BIRADS_CLF_CFG = "birads_clf_001.yml"
BMI_MODEL_CFG = "bmi_model_001.yml"
EMBEDDING_CFGS = ["tfidf_001.yml"]
SAMPLING_RATIO = 4
RAW_DATA_NAMES = ["anamnesis", "cohort", "mammogram"]

STEPS = [
    "birads-extract",
    "birads-infer",
    "preprocess",
    "find-shortcuts",
    "biopsy:get-reports",
    "biopsy:classify",
    "biopsy:get-results",
    "train-tfidf",
    "merge",
]

def run_cmd(args: List[str], dry_run: bool):
    print(">>", " ".join(args))
    if dry_run:
        return
    subprocess.run(args, check=True)

def resolve_steps(selected, run_all, step_from, step_to):
    if run_all:
        return STEPS.copy()
    if selected:
        return selected
    if step_from or step_to:
        start = STEPS.index(step_from) if step_from else 0
        end = STEPS.index(step_to) if step_to else len(STEPS) - 1
        if end < start:
            raise ValueError("--to must be after --from")
        return STEPS[start:end + 1]
    raise ValueError("Select steps with --all, --step, or --from/--to")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-dir", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--step", action="append", choices=STEPS)
    p.add_argument("--from", dest="step_from", choices=STEPS)
    p.add_argument("--to", dest="step_to", choices=STEPS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plan = resolve_steps(args.step, args.all, args.step_from, args.step_to)

    for step in plan:
        if step == "birads-extract":
            run_cmd([
                "hapcancer", "process-birads",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--process-mode", "extract",
            ], args.dry_run)

        elif step == "birads-infer":
            run_cmd([
                "hapcancer", "process-birads",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--process-mode", "infer",
            ], args.dry_run)

        elif step == "preprocess":
            for raw_name in RAW_DATA_NAMES:
                run_cmd([
                    "hapcancer", "preprocess",
                    "--config-dir", args.config_dir,
                    "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                    "--raw-data-name", raw_name,
                ], args.dry_run)

        elif step == "find-shortcuts":
            run_cmd([
                "hapcancer", "find-shortcuts",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--mode", "get-reports",
            ])

        elif step == "biopsy:get-reports":
            run_cmd([
                "hapcancer", "preprocess-biopsy",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--mode", "get-reports",
            ], args.dry_run)

        elif step == "biopsy:classify":
            run_cmd([
                "hapcancer", "preprocess-biopsy",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--mode", "classify-reports",
            ], args.dry_run)

        elif step == "biopsy:get-results":
            run_cmd([
                "hapcancer", "preprocess-biopsy",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--mode", "get-results",
            ], args.dry_run)

        elif step == "train-tfidf":
            for emb in EMBEDDING_CFGS:
                run_cmd([
                    "hapcancer", "train-tfidf",
                    "--config-dir", args.config_dir,
                    "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                    "--config-params", f"embeddings={emb}",
                    "--sampling-ratio", str(SAMPLING_RATIO),
                ], args.dry_run)

        elif step == "merge":
            run_cmd([
                "hapcancer", "merge-sources",
                "--config-dir", args.config_dir,
                "--config-params", f"birads_classifier={BIRADS_CLF_CFG}",
                "--config-params", f"bmi_model={BMI_MODEL_CFG}",
            ], args.dry_run)

if __name__ == "__main__":
    main()
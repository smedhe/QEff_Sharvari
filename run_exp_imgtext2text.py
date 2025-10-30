import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants

def slugify(name: str) -> str:
    return name.replace("/", "_")

def export_model(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "image_text_to_text"}

    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,kv_offload=True,
    )

    try:
        constants.USE_TORCH_EXPORT = True
        fx_dir = out_dir / f"{slugify(model_name)}_fx"
        fx_path = qeff_model.export(export_dir=str(fx_dir))
        results["fx_export_path"] = str(fx_path)
    except Exception as e:
        results["fx_error"] = str(e)

    return results

def main() -> int:
    models = [
        # "llava-hf/llava-1.5-7b-hf",                           #successs
        "ibm-granite/granite-vision-3.2-2b",
        # "ibm-granite/granite-vision-3.3-2b",
        # "Llama-4-Scout-17B-16E-Instruct",
        # "meta-llama/Llama-3.2-90B-Vision",
        # "llava-hf/llava-1.5-7b-hf",                           #successs
        # "meta-llama/Llama-3.2-11B-Vision-Instruct",           #successs
        # "google/gemma-3-4b-it",
        # "google/gemma-3-27b-it",
    ]

    out_dir = Path(os.environ.get("QEFF_EXPORT_OUT", "./run_exports_image")).absolute()
    hf_token = os.environ.get("HF_TOKEN")

    all_results = []
    for model_name in models:
        print(f"\n=== Processing {model_name} ===")
        res = export_model(model_name, out_dir, hf_token=hf_token)
        print(json.dumps(res, indent=2))
        all_results.append(res)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {out_dir/'summary.json'}")
    return 0

if __name__ == "__main__":
    main()
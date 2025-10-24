import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from QEfficient import QEFFAutoModel
from QEfficient.utils import constants

def slugify(name: str) -> str:
    return name.replace("/", "_")

def export_model(model_name: str, out_dir: Path, hf_token: Optional[str] = None) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": "embedding"}

    qeff_model = QEFFAutoModel.from_pretrained(
        model_name, token=hf_token, trust_remote_code=True,
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
        # "BAAI/bge-base-en-v1.5", #need seq_len max as 511 to export successfully
        # "intfloat/e5-mistral-7b-instruct", #successfully exported using torch.export
        # "sentence-transformers/multi-qa-mpnet-base-cos-v1", #successfully exported using torch.export
        # "nomic-ai/nomic-embed-text-v1.5", #not onboarded it seems
        # "NovaSearch/stella_en_1.5B_v5", #successfully exported using torch.export
        # "BAAI/bge-reranker-v2-m3", #successfully exported using torch.export
        # "ibm-granite/granite-embedding-30m-english", #need seq_len max as 513 to export successfully
        # "ibm-granite/granite-embedding-107m-multilingual", #need seq_len max as 513 to export successfully
        # "ibm-granite/granite-embedding-278m-multilingual", #need seq_len max as 513 to export successfully

    ]

    out_dir = Path(os.environ.get("QEFF_EXPORT_OUT", "./run_exports_embedding")).absolute()
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

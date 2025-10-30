import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from QEfficient import (
    QEFFAutoModel,
    QEFFAutoModelForCausalLM,
    QEFFAutoModelForImageTextToText,
)
from QEfficient.utils import constants

def slugify(name: str) -> str:
    return name.replace("/", "_")

def export_model(
    model_name: str,
    model_class: str,
    out_dir: Path,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"model": model_name, "class": model_class}
    model_cache_dir = (out_dir / "_model_cache" / slugify(model_name)).absolute()
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    qeff_model = None

    # Instantiate appropriate wrapper with per-model cache_dir
    if model_class == "causal_lm":
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, cache_dir=str(model_cache_dir), trust_remote_code=True,
        )
    elif model_class == "image_text_to_text":
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name, token=hf_token,  cache_dir=str(model_cache_dir), trust_remote_code=True,
        )
    else:
        # embedding/encoder-only
        qeff_model = QEFFAutoModel.from_pretrained(
            model_name, token=hf_token, cache_dir=str(model_cache_dir), trust_remote_code=True,
        )

    # 1) torch.export path
    try:
        constants.USE_TORCH_EXPORT = True
        fx_dir = out_dir / f"{slugify(model_name)}_fx"
        fx_path = qeff_model.export(export_dir=str(fx_dir))
        results["fx_export_path"] = str(fx_path)
    except Exception as e:
        results["fx_error"] = str(e)
    # 2) ONNX path
    try:
        constants.USE_TORCH_EXPORT = False
        onnx_dir = out_dir / f"{slugify(model_name)}_onnx"
        onnx_path = qeff_model.export(export_dir=str(onnx_dir))
        results["onnx_export_path"] = str(onnx_path)
    except Exception as e:
        results["onnx_error"] = str(e)

    return results
    # finally:
    #     # Release references and clear memory
    #     try:
    #         del qeff_model
    #     except Exception:
    #         pass
    #     gc.collect()
    #     # Delete the per-model cache directory to free disk space
    #     shutil.rmtree(model_cache_dir, ignore_errors=True)

def main() -> int:
    # Models requested (edit as needed)
    models: List[Dict[str, str]] = [
        # QEFFAutoModelForCausalLM - small models
        {"name": "google/gemma-2-9b", "class": "causal_lm"},
        {"name": "openai-community/gpt2", "class": "causal_lm"},
        {"name": "meta-llama/Llama-2-13b-chat-hf", "class": "causal_lm"},
        {"name": "OpenGVLab/InternVL2_5-1B", "class": "causal_lm"},
        # QEFFAutoModelForCausalLM - mid-size models
        {"name": "mistralai/Codestral-22B-v0.1", "class": "causal_lm"},
        {"name": "inceptionai/jais-adapted-70b", "class": "causal_lm"},
        {"name": "Qwen/Qwen3-30B-A3B-Instruct-2507", "class": "causal_lm"},
        # QEFFAutoModelForCausalLM - largest model
        {"name": "inceptionai/jais-adapted-70b", "class": "causal_lm"}, 

        # QEFFAutoModel - small models
        {"name": "BAAI/bge-base-en-v1.5", "class": "embedding"},
        {"name": "stella_en_1.5B_v5", "class": "embedding"},
        {"name": "e5-mistral-7b-instruct", "class": "embedding"},
        # QEFFAutoModel - mid-size models
        {"name": "ibm-granite/granite-embedding-30m-english", "class": "embedding"},
        {"name": "ibm-granite/granite-embedding-125m-english", "class": "embedding"},
        # QEFFAutoModel - large models
        {"name": "ibm-granite/granite-embedding-107m-multilingual", "class": "embedding"},
        {"name": "ibm-granite/granite-embedding-278m-multilingual", "class": "embedding"},

        # QEFFAutoModelForImageTextToText - small model
        {"name": "ibm-granite/granite-vision-3.2-2b", "class": "image_text_to_text"},
        # QEFFAutoModelForImageTextToText - mid-size model
        {"name": "Llama-4-Scout-17B-16E-Instruct", "class": "image_text_to_text"},
        # QEFFAutoModelForImageTextToText - large model
        {"name": "meta-llama/Llama-3.2-90B-Vision", "class": "image_text_to_text"},

        # Additional models from documentation
        {"name": "llava-hf/llava-1.5-7b-hf", "class": "image_text_to_text"},
        {"name": "meta-llama/Llama-3.2-11B-Vision-Instruct", "class": "image_text_to_text"},
        {"name": "google/gemma-3-4b-it", "class": "image_text_to_text"},
    ]

    # Output
    out_dir = Path(os.environ.get("QEFF_EXPORT_OUT", "./run_exports")).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional HF token
    hf_token = os.environ.get("HF_TOKEN")

    all_results: List[Dict[str, Any]] = []
    for spec in models:
        print(f"\n=== Processing {spec['name']} ({spec['class']}) ===")
        res = export_model(spec["name"], spec["class"], out_dir, hf_token=hf_token)
        print(json.dumps(res, indent=2))
        all_results.append(res)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {out_dir/'summary.json'}")
    return 0

if __name__ == "__main__":
    main()
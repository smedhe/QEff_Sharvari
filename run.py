from QEfficient import QEFFAutoModelForCausalLM
# from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# from QEfficient.utils import constants

# constants.USE_TORCH_EXPORT=True
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

model_name = "Snowflake/Llama-3.1-SwiftKV-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# print("tokenizer load done")
model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=False)
print("model load done")

model.export()
print("run model done")
# model.compile(num_cores=16, batch_size = 1, ctx_len=1024,prefill_seq_len=128, 
#                 mxfp6_matmul=True, mxint8_kv_cache=True, num_devices=4, aic_enable_depth_first=True, 
#                 allow_mxint8_mdp_io=True)
# compile_end = time.time()

# print(f"compile done in {compile_end-end} sec")
# ep = torch.export.load(export_path)

# print(ep.graph_module.code)

# prompts = ["what is docker", "Write an essay about engineers","how to write a program to calculate the sum of two numbers"]
# tokenizer.pad_token = tokenizer.eos_token
# text = tokenizer(prompts, return_tensors = "pt", padding="longest")
# # Extract sequence length from attention_mask
# seq_len = text["attention_mask"].size(1)

# # Create position_ids tensor: shape [1, seq_len], values [0, 1, ..., seq_len - 1]
# position_ids = torch.arange(seq_len).unsqueeze(0)

# # Optionally, add it to the inputs dictionary if your model expects it
# text["position_ids"] = position_ids
# epm = torch.export.load(export_path)
# output = epm.module()(**text)
# print("run model done")


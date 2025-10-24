# import pdb; pdb.set_trace()
from transformers import AutoTokenizer
# from QEfficient.utils.logging_utils import QEFFLogger
# logger = QEFFLogger.get_logger(namespace="INFRA", loglevel="INFO")
from QEfficient import QEFFAutoModelForCausalLM
import time
from tabulate import tabulate
 
model_name="meta-llama/Llama-3.1-70B"
input="Help me with this"
tokenizer = AutoTokenizer.from_pretrained(model_name)
t1=time.time()
model=QEFFAutoModelForCausalLM.from_pretrained(model_name)
# logger.info(f"Executing {model_name} with Modeling class {model.__class__}")
t2=time.time()
model.export()
t3=time.time()
print(f"time for model onnx export", t3-t2)
model.compile(num_devices=1,num_cores=16 ) # Considering you have a Cloud AI 100 Standard SKU
t4=time.time()
print(f"time for onnx export compile", t4-t3)
# print(model.generate(prompts=["write a haiku about sun and moon"], tokenizer=tokenizer))
# t5=time.time()
 
# Replace these with your actual timing values
timing_data = [
    ["Model Loading",  t2 - t1],
    ["Model Exporting", t3 - t2],
    ["Model Compilation", t4 - t3],
]
 
# Print the table
print(tabulate(timing_data, headers=["Step", "Time (s)"], tablefmt="github"))
 
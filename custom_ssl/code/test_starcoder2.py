from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

t1= time.perf_counter()
model_name = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

t2= time.perf_counter()
print(f"Loading tokenizer and model: took {t2-t1} seconds to execute.")
# Create a pipeline
code_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

t3= time.perf_counter()
print(f"Creating piepline: took {t3-t2} seconds to execute.")
# Generate code for an input string
while True:
  print("\n=========Please type in your question=========================\n")
 
  user_content =  [{f'role': 'user', 
            f'content': 
            #f"{data[0]}" 
            f"Can you tell me the time complexity of the code based on "
            f"\n1. O(1) \n2. O(log n) \n3. O(n) \n4. O(n log n) \n5. O(n^2) \n6. O(n^3) \n7. O(2^n)?\n"
            f"Say something like, “**The time complexity of this code is (time complexity of code)."    }]
  
  
  #user_content.strip()
  t1= time.perf_counter()
  generated_code = code_generator(user_content, max_length=256)[0]['generated_text']
  t2= time.perf_counter()
  print(f"Inferencing using the model: took {t2-t1} seconds to execute.")
  print(generated_code)
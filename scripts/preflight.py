import os,torch,importlib.metadata as m
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
assert torch.cuda.is_available(),'CUDA GPU required'
for k in ['torch','transformers','peft','liger-kernel']:print(k,m.version(k))
for i in range(torch.cuda.device_count()):print(i,torch.cuda.get_device_name(i),round(torch.cuda.get_device_properties(i).total_memory/2**30,1),'GiB')
p=os.environ['MODEL_PATH'];t=AutoTokenizer.from_pretrained(p,local_files_only=True);s=t.apply_chat_template([{'role':'user','content':'test'}],tokenize=False,add_generation_prompt=True,enable_thinking=True);assert s.endswith('<think>\n');print('Tokenizer and thinking prefix OK')

import locale
locale.getpreferredencoding = lambda: "UTF-8"
from torch import cuda, bfloat16

import torch

import transformers


model_id = '' #enter the path to your llama model here
# it generally looks like this: r"C:\Users\<your user name>\<directory name of virtual environment>\env\Lib\site-packages\transformers\models\llama\llama-2-7b-chat-hf"

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True,
    bnb_4bit_compute_dtype = bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code= True, 
    quantization_config = bnb_config,
    device_map = 'cuda:0'
)

###
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
)

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]


import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    temperature=0.8,
    max_new_tokens=256,
    repetition_penalty=1.0
)

def gen_txt(prompt):
    try:
        res = generate_text(prompt)
        print('Machine:',res[0]["generated_text"][len(prompt):])
    except:
        pass


print('llama File done')

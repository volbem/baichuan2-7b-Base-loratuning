from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import torch
import argparse
from transformers import set_seed
from accelerate import Accelerator
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import set_seed
import random
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path,tokenizer,max_length=900):
        self.dataset = {}
        with open(data_path,encoding='utf-8',mode='r') as f:
            self.dataset = json.load(f)
        self.datas = []
        self.tokenizer = tokenizer
        self.user_token = self.tokenizer.convert_ids_to_tokens(195)
        self.assistant_token = self.tokenizer.convert_ids_to_tokens(196)
        self.max_length=max_length
        for index,sample in enumerate(self.dataset):
            da={}
            da['input_token'], da["reference"] = self.generate_prompt(sample["conversations"])
            self.datas.append(da)

    def generate_prompt(self,conversation):
        input_token=""
        for message in conversation[:-1]:
            from_ = message["from"]
            value = message["value"]
            if from_ == "human":
                input_token += self.user_token + value
            else:
                input_token += self.assistant_token + value
        input_token += self.assistant_token
        reference_token = conversation[-1]["value"]
        return input_token, reference_token
        


    def __getitem__(self, index):
        da = self.datas[index]
        return {
            'reference': da['reference'],
            'input': da['input_token']
        }
    
    def __len__(self):
        return len(self.datas)
    
    def collate_fn(self, batch):
        batch_input = [x['input'] for x in batch]
        batch_reference = [x['reference'] for x in batch]
        out_batch = {}
        out_batch['reference'] = batch_reference
        out_batch['input'] = batch_input
        output_tokenizer = self.tokenizer(batch_input, return_tensors='pt', padding='longest')
        out_batch['input_ids'] = output_tokenizer['input_ids']
        out_batch['attention_mask'] = output_tokenizer['attention_mask']
        if output_tokenizer['input_ids'].shape[-1] > self.max_length:
            out_batch['input_ids'] = out_batch['input_ids'][:,-self.max_length:]
            out_batch['attention_mask'] = out_batch['attention_mask'][:,-self.max_length:]
        return out_batch



def get_response(inputs,outputs,tokenizer,num_return=1):
    responses_list=[]
    batch_return=[]
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i%num_return==num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def test(args):
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.process_index)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.cuda().eval()
    accelerator.print(f'args:\n{args}')
    left_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='right')
    left_tokenizer.pad_token_id = 0
    accelerator.print(f'load_finish')
    dataset = TestDataset(args.data_path,left_tokenizer)
    # 注意batch_size
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)
    args.num_return = 1
    gen_kwargs = {'num_return_sequences': args.num_return, 'max_new_tokens': args.max_new_tokens}
    val_dataloader = accelerator.prepare(val_dataloader)
    accelerator.wait_for_everyone()
    
    cache_reference = []
    cache_response = []
    cache_input=[]
    with torch.no_grad():
        ress = []
        dataloader_iterator = tqdm(val_dataloader, total=len(val_dataloader)) if accelerator.is_main_process else val_dataloader
        for batch in dataloader_iterator:
            input_ids = batch["input_ids"]
            reference = batch["reference"]
            input = batch["input"]
            attention_mask = batch["attention_mask"]
            outputs = model.generate(input_ids, attention_mask=attention_mask,**gen_kwargs)
            response = get_response(input_ids,outputs,left_tokenizer,args.num_return)
            cache_reference.extend(reference)
            cache_response.extend(response)
            cache_input.extend(input)

        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
        all_reference =  [None] * dist.get_world_size()
        all_response =  [None] * dist.get_world_size()
        all_input =  [None] * dist.get_world_size()
        dist.all_gather_object(all_response,cache_response)
        dist.all_gather_object(all_reference,cache_reference)
        dist.all_gather_object(all_input,cache_input)
        all_reference = [item for sublist in all_reference for item in sublist]
        all_response = [item for sublist in all_response for item in sublist]
        all_input = [item for sublist in all_input for item in sublist]
        for reference, response,input in zip(all_reference,all_response,all_input):
            d={}
            d["reference"] = reference
            d["response"] = response
            d["input"] = input
            ress.append(d)

        if accelerator.is_main_process:
            print(f'test results: {args.out_file}')
            outstr = json.dumps(ress,ensure_ascii=False,indent = 2)
            with open(args.out_file,'w', encoding='utf-8') as f:
                f.write(outstr)

    # wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')

    # Model Args
    parser.add_argument('--data_path', default='./data/kst_test_original.json', type=str)
    parser.add_argument('--model_path', default='../Baichuan2/Baichuan2-7B-Base', type=str)
    # Other Args
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_new_tokens', default=128, type=int)
    parser.add_argument('--out_file', default='./results_original/output.json', type=str)
    args = parser.parse_args()
    set_seed(args.seed)
    test(args) 




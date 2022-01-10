import argparse
import tqdm
import numpy as np
from pathlib import Path
from smart_open import open as sopen
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, BertForMaskedLM, AutoModel

from dataset_reader import TextDatasetReader, InputFeatures
from adaptive_sampler import (
    MaxTokensBatchSampler,
    data_collator_for_adaptive_sampler,
)
#from subs_cleaner import clean_subs
#from special_tokens import SpecialTokens
#from utils import NUM_SUBS_TO_USE, SUBSTITUTES_DIR, find_ann_path, try_until_success_or_limit
import ray
from google.cloud import storage
from smart_open import open as sopen # type: ignore
import time

class Encoder(object):
    """
    A wrapper over a torch model
    """
    def __init__(self, model):
        self.model = model
    
    def get_hidden_states(self, inputs: InputFeatures):
    
        guids = inputs.pop("guid")
        dict_to_device(inputs, self.model.device)
        last_hidden_states = self.model(**inputs)[0]
        return last_hidden_states
                        
    def get_vector(self, inputs: InputFeatures) -> torch.Tensor:
    
        """ return a vector representation of the sentence """
        hidden_states = self.get_hidden_states(inputs)
        return self.extract_vector_from_hidden_states(hidden_states)
    
    def extract_vector_from_hidden_states(self, hidden_states: torch.Tensor) -> np.ndarray:
    
        raise NotImplementedError
        

class MeanEncoder(Encoder):

    def __init__(self, model):
        super().__init__(model)
        
    def extract_vector_from_hidden_states(self, hidden_states):
    
        mean = hidden_states.mean(dim=1)
        return mean.detach().cpu().numpy()
        
class FirstTokenEncoder(Encoder):

    def __init__(self, model):
        super().__init__(model)
        
    def extract_vector_from_hidden_states(self, hidden_states):
    
        return hidden_states[:,0,:].detach().cpu().numpy()


def process_dataset(encoder: Encoder, input_path: str, tokenizer, args):
        dataset = TextDatasetReader(input_path, tokenizer)
        dataloader = adaptive_dataloader(args, dataset)
        
        all_states = []
        output_path = input_path.replace("/text/", "/states/").replace(".jsonl.gz", ".npy")
        with sopen(output_path, "wb") as outfh:
            for inputs in dataloader:
                H = encoder.get_vector(inputs)
                all_states.append(H)

            all_states = np.concatenate(all_states, axis = 0)
            print("saving states of shape {} in {}".format(all_states.shape, output_path))
            np.save(outfh, all_states)

@ray.remote
def process_dataset_ray(encoder: Encoder, input_path: str, tokenizer, args):
    return process_dataset(encoder, input_path, tokenizer, args)
                        
def main(args, fnames):

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer, model = initialize_models(device, args)    
    encoder = FirstTokenEncoder(model)
    tasks = []
    
    i = 0
    for fname in fnames:
        #process_dataset(encoder, dataloader,fname)
        tasks.append(process_dataset_ray.remote(encoder, fname, tokenizer, args))
        i += 1
        if i > 10: 
            break
            
    start = time.time()
    results = ray.get(tasks)
    print("running time", time.time() - start)            
          
"""
def main(args):
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    tokenizer, model = initialize_models(device, args)
    encoder = FirstTokenEncoder(model)
    dataset = TextDatasetReader(args, input_file, tokenizer)
    
    
    files = find_ann_path(args)
    for input_file in tqdm(files):
        print("Running on input file: ", input_file)
        dataset = TextDatasetReader(args, input_file, tokenizer)
        if not dataset:
            print("Skipping empty file")
            continue
        dataloader = adaptive_dataloader(args, dataset)

        batch_idx = 0
        for inputs in dataloader:
            with torch.no_grad():
                guids = inputs.pop("guid")
                dict_to_device(inputs, device)
                last_hidden_states = model(**inputs)[0]
                _, substitutes = last_hidden_states.topk(100)
                substitutes = clean_subs(special_tokens, inputs["input_ids"], substitutes.cpu())
                write_substitutes_to_file(
                    infer_outfile(args.output_path, input_file, batch_idx), guids, inputs, substitutes
                )
            batch_idx += 1
"""

def infer_outfile(output_dir, input_file, input_idx):
    filename = input_file[input_file.rfind("/") + 1 :]
    return output_dir + f"/{SUBSTITUTES_DIR}/" + filename + f"-{input_idx}"


def write_substitutes_to_file(outfile, sent_guids, inputs, substitutes):
    attention_mask = inputs["attention_mask"].bool()

    sent_lengths = inputs["attention_mask"].sum(1)
    tokens = inputs["input_ids"].masked_select(attention_mask)
    substitutes = substitutes.masked_select(attention_mask.cpu().unsqueeze(2)).view(-1, NUM_SUBS_TO_USE)

    write_results(outfile, tokens, sent_lengths, substitutes, sent_guids)


def write_results(outfile, tokens, sent_lengths, substitutes, sent_guids):
    def _write_results(file):
        np.save(sopen(f"{file}-tokens.npy", "wb"), tokens.cpu().numpy().astype(np.uint16))
        np.save(sopen(f"{file}-lengths.npy", "wb"), sent_lengths.cpu().numpy().astype(np.int16))
        np.save(sopen(f"{file}-subs.npy", "wb"), substitutes.cpu().numpy().astype(np.uint16))
        np.save(sopen(f"{file}-sent_guids.npy", "wb"), np.array(sent_guids, dtype=object))

    try_until_success_or_limit(_write_results, outfile)


def dict_to_device(inputs, device):
    if device.type == "cpu":
        return
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)


def initialize_models(device, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    if args.fp16:
        from apex import amp
        model = amp.initialize(model, opt_level="O2")

    assert tokenizer.vocab_size < 65535  # Saving pred_ids as np.uint16
    return tokenizer, model


def adaptive_dataloader(args, dataset):
    sampler = MaxTokensBatchSampler(dataset, max_tokens=args.max_tokens_per_batch, padding_noise=0.0)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=data_collator_for_adaptive_sampler,)
    return dataloader

def collect_paths(directory):
    client = storage.Client()
    bucket = client.get_bucket("ai2i-us")
    fnames = []

    i = 0
    for blob in bucket.list_blobs(prefix='SPIKE/datasets/text/{}/'.format(directory)):
        if blob.name.endswith(".jsonl.gz"):
            fnames.append("gs://ai2i-us/"+blob.name)
            i+=1
            if i > 10: break
    return fnames 
    
def adaptive_dataloader(args, dataset):
    sampler = MaxTokensBatchSampler(dataset, max_tokens=args.max_tokens_per_batch, padding_noise=0.0)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=data_collator_for_adaptive_sampler,)
    return dataloader
    
        
if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dir', type=str, default="wikipedia",
                    help='directory name within https://console.cloud.google.com/storage/browser/ai2i-us/SPIKE/datasets/text/')
    arg_parser.add_argument(
        "-o", "--output_path", help="where to write files. On cloud will be best", type=str, required=True
    )
    arg_parser.add_argument("--data_dir", help="path to where run files are stored.", type=Path, default="~/wsi")
    arg_parser.add_argument("--max_tokens_per_batch", type=int, required=True)
    arg_parser.add_argument("--input_suffix", help="glob pattern to use.", type=str, default="*.gz")
    arg_parser.add_argument(
        "--split_files_into_n_part",
        help="If specified, we will break the " "files list into few splits and process them in different runs",
        type=int,
    )
    arg_parser.add_argument(
        "--files_split_to_process",
        help="If specified, will process just the files_split_to_process part of files."
        "This is 1-indexing, e.g., there are 100 files, numbered 0-99, split_files_into_n_part=10, "
        "files_split_to_process=6, we'll use the 60th-69th files.",
        type=int,
    )
    arg_parser.add_argument("--force_cpu", action="store_true", help="Should force cpu even when a gpu is available")
    arg_parser.add_argument("--model_name", type=str, help="huggingface model name", default="bert-base-uncased")
    arg_parser.add_argument("--fp16", action="store_true", help="If specified, use fp16.")
    arg_parser.add_argument("--cuda_device", type=int, choices=[0, 1, 2, 3], default=0)
    
    args = arg_parser.parse_args()
    fnames = collect_paths(args.dir)
    """
    if args.split_files_into_n_part:
        assert args.files_split_to_process and (args.files_split_to_process - 1) < args.split_files_into_n_part

    if not args.output_path.startswith("gs://"):
        Path(args.output_path + f"/{SUBSTITUTES_DIR}").mkdir(exist_ok=True)
    
    print(args)
    """
    ray.shutdown()
    #ray.init(local_mode=True)
    ray.init(address="auto")
    main(args, fnames)

import argparse
import time

import numpy as np
import ray
import torch
from google.cloud import storage
from smart_open import open as sopen  # type: ignore
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer

from adaptive_sampler import (MaxTokensBatchSampler,
                              data_collator_for_adaptive_sampler)
from dataset_reader import InputFeatures, TextDatasetReader


class Encoder(object):
    """
    A wrapper over a torch model
    """
    def __init__(self, model):
        self.model = model

    def get_hidden_states(self, inputs: InputFeatures):
        with torch.no_grad():
            dict_to_device(inputs, self.model.device)
            last_hidden_states = self.model(**inputs)[0]
        return last_hidden_states

    def get_vector(self, inputs: InputFeatures) -> torch.Tensor:
        """ return a vector representation of the sentence """
        sentence_ids = inputs.pop("guid")
        hidden_states = self.get_hidden_states(inputs)
        return sentence_ids, self.extract_vector_from_hidden_states(hidden_states)

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
    all_sentence_ids = []
    for inputs in dataloader:
        sentence_ids, H = encoder.get_vector(inputs)
        all_states.append(H.copy())
        all_sentence_ids.append(sentence_ids.copy())

    all_states = np.concatenate(all_states, axis = 0)
    all_sentence_ids = np.concatenate(all_sentence_ids, axis = 0)

    write_states(input_path, all_states)
    write_sent_ids(input_path, all_sentence_ids)
    # MAYBE: is this really needed?
    del all_states
    del all_sentence_ids

    print("done with file", input_path)

@ray.remote(num_gpus=1)
def process_dataset_ray(encoder: Encoder, input_path: str, tokenizer, args):
    process_dataset(encoder, input_path, tokenizer, args)

def main(args, fnames):

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("device:", device)
    tokenizer, model = initialize_models(device, args)
    encoder = FirstTokenEncoder(model)
    tasks = []

    for fname in fnames:
        tasks.append(process_dataset_ray.remote(encoder, fname, tokenizer, args))

    start = time.time()
    ray.get(tasks)
    print("running time", time.time() - start)

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

def collect_paths(directory, skip_done):
    client = storage.Client()
    bucket = client.get_bucket("ai2i-us")
    fnames = []

    done_files = set(
        ["gs://ai2i-us/"+blob.name
            for blob in bucket.list_blobs(prefix='SPIKE/datasets/states/{}/'.format(directory))
            if "sent_ids" in blob.name]
    )

    for blob in bucket.list_blobs(prefix='SPIKE/datasets/text/{}/'.format(directory)):
        if blob.name.endswith(".jsonl.gz"):
            filename = "gs://ai2i-us/"+blob.name
            if skip_done and store_filename(filename, "sent_ids", ext="txt.gz") in done_files:
                continue
            fnames.append(filename)
    print(f"found {len(fnames)} fnames")
    return fnames

def adaptive_dataloader(args, dataset):
    sampler = MaxTokensBatchSampler(dataset, max_tokens=args.max_tokens_per_batch, padding_noise=0.0)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=data_collator_for_adaptive_sampler,)
    return dataloader

def write_states(input_path, all_states):
    np.save(sopen(store_filename(input_path, "states"), "wb"), all_states)

def write_sent_ids(input_path, all_sentence_ids):
    with sopen(store_filename(input_path, "sent_ids", ext="txt.gz"), "w") as f:
        f.write("\n".join(all_sentence_ids.tolist()))

def store_filename(input_path, object_name, ext="npy.gz"):
    return input_path.replace("/text/", "/states/").replace(".jsonl.gz", f"-{object_name}.{ext}")

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dir', type=str, default="wikipedia",
                    help='directory name within https://console.cloud.google.com/storage/browser/ai2i-us/SPIKE/datasets/text/')
    arg_parser.add_argument("--max_tokens_per_batch", type=int, required=True)
    arg_parser.add_argument("--force_cpu", action="store_true", help="Should force cpu even when a gpu is available")
    arg_parser.add_argument("--model_name", type=str, help="huggingface model name", default="bert-base-uncased")
    arg_parser.add_argument("--fp16", action="store_true", help="If specified, use fp16.")
    arg_parser.add_argument("--skip_done", action="store_true", help="Skip files already processed.")
    arg_parser.add_argument("--cuda_device", type=int, choices=[0, 1, 2, 3], default=0)

    args = arg_parser.parse_args()
    fnames = collect_paths(args.dir, args.skip_done)

    ray.shutdown()
    ray.init(address="auto")
    main(args, fnames)

from __future__ import annotations
import sys
import argparse
from types import SimpleNamespace
from typing import List, IO, Optional, Any, Tuple, Union
import logging
import time
from pathy import Pathy
from pathlib import Path
import sys

import ray
from smart_open import open as sopen
import os
import json
import tqdm
import ray
from google.cloud import storage
import argparse

def recursive_collect_sents(d: dict):
    
    all_sents, all_ids = [], []
    
    if "sentences" in d:
        sents = d["sentences"]
        for s in sents:
            sent_id, txt = s["sentenceId"], " ".join(s["words"]).replace("\n","")
            all_ids.append(sent_id)
            all_sents.append(txt)
    
    else:
            for inner_d in d["body"]:
                inner_sents, inner_ids = recursive_collect_sents(inner_d)
                all_sents.extend(inner_sents)
                all_ids.extend(inner_ids)
    
    
    return all_sents, all_ids

def process_file(in_file_name, out_file_name):
    count = 0
    with sopen(in_file_name) as infh:
        with sopen(out_file_name, "w") as outfh:
            for line in infh:
                d = json.loads(line)
                count +=1
                sents,ids = recursive_collect_sents(d)
                for txt,sent_id in zip(sents, ids):
                    outfh.write(json.dumps({"text": txt, "sent_id": sent_id}) + "\n")
                    outfh.write(json.dumps({"text": txt, "sent_id": sent_id}) + "\n")
    return 1


@ray.remote
def process_file_ray(in_file_name, out_file_name):
    return process_file(in_file_name, out_file_name)

def collect_paths(directory):
    client = storage.Client()
    bucket = client.get_bucket("ai2i-us")
    fnames = []

    for blob in bucket.list_blobs(prefix='SPIKE/datasets/annh/{}/'.format(directory)):
        if blob.name.endswith(".jsonl.gz"):
            fnames.append("gs://ai2i-us/"+blob.name)
    return fnames 
      
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='extracts sents+ids from annh files.')
    parser.add_argument('--dir', type=str, default="wikipedia",
                    help='directory name in https://console.cloud.google.com/storage/browser/ai2i-us/SPIKE/datasets/annh/')

    args = parser.parse_args()
    
    ray.shutdown()
    #ray.init(local_mode=True)
    ray.init(address="auto")
    
    # collect tasks
    
    tasks = []
    fnames = collect_paths(args.dir)
    start = time.time()
    i = 0
    for in_f in fnames:
        out_f = in_f.replace("/annh/","/text/")
        tasks.append(process_file_ray.remote(in_f, out_f))
        i += 1      
        #if i > 10: break
    print("collection time", time.time() - start)    

    # run tasks
    
    start = time.time()
    results = ray.get(tasks)
    print("running time", time.time() - start)

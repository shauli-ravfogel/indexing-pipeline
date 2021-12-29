import json
import dataclasses
from dataclasses import dataclass
from typing import List, Union, Optional, Iterator, Tuple
from pathy import Pathy
import logging
import os
import time

import torch
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import DataProcessor, InputExample
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import AutoTokenizer, BertForMaskedLM

#from utils import try_until_success_or_limit
from smart_open import open as sopen # type: ignore
import tqdm

logger = logging.getLogger(__name__)

MAX_LENGTH = 512


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    guid: Optional[Union[int, float, str]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"



class TextDatasetReader(Dataset):
    def __init__(
        self, input_path: str, tokenizer: PreTrainedTokenizer, cache_dir: Optional[str] = None,
    ):
    
        """
        self.processor = AnnProcessor()
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else "/tmp/",
            "cached_{}_{}".format(tokenizer.__class__.__name__, os.path.basename(input_file)),
        )
        
        if os.path.exists(cached_features_file):
            start = time.time()
            self.features = torch.load(cached_features_file)
            logger.info(f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)
        else:
        
            examples = list(self.processor.get_examples(input_file))
            if not examples:
                self.features = []
                return
            self.features = convert_examples_to_features(
                examples, tokenizer, max_length=MAX_LENGTH, padding_strategy="do_not_pad"
            )
            start = time.time()
            torch.save(self.features, cached_features_file)
            logger.info("Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start)
        """
        
        examples = []
        with sopen(input_path) as infh:
            for line in infh:
                d = json.loads(line)
                example = InputExample(guid=d["sent_id"], text_a=d["text"])
                examples.append(example)
                
        self.features = convert_examples_to_features(
                examples, tokenizer, max_length=MAX_LENGTH, padding_strategy="do_not_pad"
            )
        #start = time.time()
        #torch.save(self.features, cached_features_file)
        #logger.info("Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, x) -> Union[InputFeatures, List[InputFeatures]]:
        if isinstance(x, list):
            return [self.features[i] for i in x]
        return self.features[x]
        
def convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding_strategy: str = "max_length",
):
    if max_length is None or max_length == -1:
        max_length = tokenizer.max_len

    batch_encoding = tokenizer(
        [example.text_a for example in examples],
        max_length=max_length,
        padding=padding_strategy,
        truncation=True,
        add_special_tokens=True,
    )

    features = []
    for inputs, guid in merge_encodings(batch_encoding, examples):
        feature = InputFeatures(**inputs, guid=guid)
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def merge_encodings(encoding, examples):
    lengths = [len(x) for x in encoding["input_ids"]]

    batch_length = 0
    concat_encoding = None
    for i, length in enumerate(lengths):
        end_of_batch = False
        batch_length += length
        curr_guid = examples[i].guid
        if batch_length > MAX_LENGTH:
            end_of_batch = True
        else:
            if (
                len(lengths) == i + 1
                or curr_guid != examples[i + 1].guid
                or (batch_length + lengths[i + 1]) > MAX_LENGTH
            ):
                end_of_batch = True

        if concat_encoding is None:
            concat_encoding = {"input_ids": encoding["input_ids"][i], "attention_mask": encoding["attention_mask"][i]}
        else:
            concat_encoding = {k: concat_encoding[k] + encoding[k][i] for k in concat_encoding}

        if end_of_batch:
            yield concat_encoding, curr_guid
            batch_length = 0
            concat_encoding = None
            
if __name__ == "__main__":
    from google.cloud import storage
    
    def collect_paths(directory):
        client = storage.Client()
        bucket = client.get_bucket("ai2i-us")
        fnames = []

        for blob in bucket.list_blobs(prefix='SPIKE/datasets/text/{}/'.format(directory)):
            if blob.name.endswith(".jsonl.gz"):
                fnames.append("gs://ai2i-us/"+blob.name)
        return fnames 
    
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    paths = collect_paths("wikipedia")
    TextDatasetReader(paths[0], tokenizer)

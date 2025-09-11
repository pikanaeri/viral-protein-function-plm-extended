from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import time
import gc

from tqdm import tqdm
import pickle
from typing import List
import numpy as np

### TO DO: conver the logging steps to Value Exceptions

def _embed_seqs(transformer: T5EncoderModel, tok: T5Tokenizer, sequences: List[str], batch_size: int) -> np.ndarray:
    vectors = np.empty(shape = (0,1024), dtype=np.float32)
    emb = transformer.compute_embeddings(sequences, pool_mode=('mean'), batch_size=batch_size)
    vectors = np.concatenate((vectors, emb['mean']), axis=0)
    
    return vectors

def _get_faa(path: str, max_length: int = 0) -> List[str]:
    idents = []
    seqs = []
    seq = []

    with(open(path)) as file:
        for line in file:
            line = line.rstrip()
            if line.startswith('>'):
                idents.append(line)
                if len(seq) > 0:
                    seqs.append(''.join(seq).replace('-', ''))
                    seq = []
            else:
                seq.append(line)
    seqs.append(''.join(seq).replace('-', ''))

    # protbert_bfd can only handle sequences < 5096aa
    if max_length > 0:
        seqs = [x[0:max_length] for x in seqs]

    return idents, seqs

def prott5_xl_uniref50_embed(faa_path: str, max_length: int, num_gpus: int, batch_size: int) -> dict:

    identifiers, sequences = _get_faa(faa_path, max_length=max_length)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') ## ProtT5 only appears to support 1 GPU, in half precision
    print("Using device: {}".format(device))
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    print("Loading: {}".format(transformer_link))
    model = T5EncoderModel.from_pretrained(transformer_link)
    model.full() if device=='cpu' else model.half()
    model = model.to(device)
    model = model.eval()
    
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False )

    ## batch sequence embedding to reduce memory
    d = {}
    sequence_batch = 100
    
    for i in range(int(len(sequences)/sequence_batch)+1):

        start = i*sequence_batch
        end = (i+1)*sequence_batch
        ## account for instance when there is no remainder
        if start == len(sequences):
            continue

        s_vectors = _embed_seqs_prott5(transformer=model, tok=tokenizer, sequences=sequences[start:end], batch_size=batch_size)
        d.update(dict(zip(identifiers[start:end], s_vectors)))


    return d

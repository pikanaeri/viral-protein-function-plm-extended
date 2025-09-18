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

def _embed_seqs_prott5(transformer: T5EncoderModel, tok: T5Tokenizer, sequences: List[str], batch_size: int) -> np.ndarray:
    ## code from https://github.com/agemagician/ProtTrans/tree/master
    sequences = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tok(sequence_examples, add_special_tokens=True, padding="longest")
    
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for id in range(gsz):
      emb = embedding_repr.last_hidden_state[id, :len(sequence_examples[id])]
      emb = emb.mean(dim=0)
      emb = emb.cpu().numpy()
      vectors.append(np.array(emb))
    
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

    ## code from https://github.com/agemagician/ProtTrans/tree/master
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    
    # Load the model
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc").to(device)
    
    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    if device == torch.device("cpu"):
        model.to(torch.float32)

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

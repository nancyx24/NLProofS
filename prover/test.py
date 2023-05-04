import sys
sys.path.append('../')
from copy import deepcopy
from common import *
from prover.proof import Proof, InvalidProofStep
import random
import json
import itertools
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

def read_street_proofs(path: str, is_train: bool, dataset: str):
    """
    Load the STREET data sets. Dataset parameter "gsm8k" or "scone".
    """
    data = []
    num_invalid = 0
    for line in open(path):
        ex = json.loads(line)
        # equivalent to context for Entailment Bank
        linearized_input = ex['linearized_input']
        context = extract_context(linearized_input)
        # equivalent to proof_text for Entailment Bank
        linearized_output = ex['linearized_output']
        proof_text = normalize(linearized_output.strip())
        # equivalent to hypothesis for Entailment Bank
        hypothesis = normalize('The answer is ' + str(ex['answer']))
        proof_text = re.sub(r'int.: ' + hypothesis, 'hypothesis', proof_text)

        if 'hypothesis' not in proof_text:
            num_invalid += 1
            continue
        
        try:
            proof = Proof(
                context,
                hypothesis,
                proof_text,
                strict=is_train,
                requires_complete=is_train,
            )
            data.append({"proof": proof})
        except InvalidProofStep:
            assert is_train
            num_invalid += 1

    print(f"{len(data)} proofs loaded. {num_invalid} invalid ones removed.")

    return data

read_street_proofs("../data/gsm8k/reasoning_annotated_dev.jsonl" ,False, "gsm8k")

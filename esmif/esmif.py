import torch
import torch.nn.functional as F
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--r1", type=int, required=True)
parser.add_argument("--r2", type=int, required=True)
parser.add_argument("--tsv", type=str, required=True)
parser.add_argument("--save", type=str, required=True)
parser.add_argument("--pdb", type=str, required=True)
args = parser.parse_args()

pdb_file = args.pdb
chain = "A"

logits = None


@torch.no_grad()
def score_sequence(model, alphabet, coords, seq):
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch)

    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx).to("cuda")

    global logits
    if logits is None:
        logits, _ = model.forward(
            coords.to("cuda"),
            padding_mask.to("cuda"),
            confidence.to("cuda"),
            prev_output_tokens.to("cuda")
        )
    else:
        pass

    loss = F.cross_entropy(logits, target.to("cuda"), reduction='none')

    avgloss = torch.sum(loss * ~target_padding_mask, dim=-1) / torch.sum(~target_padding_mask, dim=-1)
    ll_fullseq = -avgloss.detach().cpu().numpy().item()

    coord_mask = torch.all(torch.all(torch.isfinite(coords.to("cuda")), dim=-1), dim=-1)
    coord_mask = coord_mask[:, 1:-1]
    avgloss = torch.sum(loss * coord_mask, dim=-1) / torch.sum(coord_mask, dim=-1)
    ll_withcoord = -avgloss.detach().cpu().numpy().item()

    return ll_fullseq, ll_withcoord


model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.cuda()
model = model.eval()
coords, seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
print(seq)
import pandas as pd
from tqdm import tqdm


def full_sequence(origin_sequence, raw_mutant):
    list_mutants = raw_mutant.split(";")
    sequence = origin_sequence
    for raw_mut in list_mutants:
        to = raw_mut[-1]
        pos = int(raw_mut[1:-1]) - 1
        if sequence[pos] != raw_mut[0]:
            raise ValueError("Fuck")
            # f"{raw_mut} the original sequence is {pos}-{sequence[pos]} different to that in the mutant file in resid  {raw_mut[0]}"
        sequence = sequence[:pos] + to + sequence[pos + 1:]
    return sequence


df = pd.read_table(args.tsv)
df = df.iloc[args.r1:args.r2]

ifscore = []
for m in tqdm(df.mutant):
    try:
        mseq = full_sequence(seq, m)
    except ValueError:
        print(m)
    ll, _ = score_sequence(model, alphabet, coords, mseq)
    ifscore.append(ll)

df['ifscore'] = ifscore

df.to_csv(args.save, sep="\t", index=False)

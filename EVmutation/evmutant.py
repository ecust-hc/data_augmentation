import pandas as pd
from model import CouplingsModel
import tools
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of EVmutantion models."  
    ) 
    parser.add_argument(
        "--model-params",
        type=str,
        help="location of model file .example:example/PABP_YEAST.model_params",
    )
    parser.add_argument(
        "--mutantion",
        type=str,
        help="location of mutantion",
    )
    parser.add_argument(
        "--computer-mode",
        type=str,	
        help="single or multiple",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="result location",
    )
    
    return parser

def extract_mutations(mutation_string, offset=0):
    if mutation_string.lower() not in ["wild", "wt", ""]:
        mutations = mutation_string.split(";")
        return list(map(lambda x: (int(x[1:-1]) + offset, x[0], x[-1]),mutations))
    else:
        return []

def main(args):
    
    # load parameters from file to create a pairwise model
    c = CouplingsModel(args.model_params)
    # read the experimental mutational scanning dataset for PABP by Melamed et al., RNA, 2013
    data = pd.read_csv(
        args.mutantion, sep=";", comment="#"
    )
    if args.computer_mode == "single":
	# predict mutations using our model
        data_pred = tools.predict_mutation_table(
            c, data, "effect_prediction_epistatic"
        )
        data_pred.to_csv(args.save, sep="\t", index=False)
    else:
        evscore=[]
        for m in data.mutant:
            mutant = extract_mutations(m)
			
            # double mutant L186M, G188A
            delta_E, delta_E_couplings, delta_E_fields = c.delta_hamiltonian(mutant)
            evscore.append(delta_E)
        data["evscore"] = evscore 
        data.to_csv(args.save, sep="\t", index=False)
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

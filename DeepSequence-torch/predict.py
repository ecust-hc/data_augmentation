from model import VariationalAutoencoder, load_model
from utils import DataHelper
from predictor import MutationEffectPredictor
import torch
import argparse
from tqdm import tqdm
import pandas as pd

def main(args):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    data = DataHelper(alignment_file=args.alignment_file, calc_weights=False)

    vae_model = VariationalAutoencoder(data.seq_len, data.alphabet_size, latent_dim=30,
                                       enc_h1_dim=1500, enc_h2_dim=1500,
                                       dec_h1_dim=100, dec_h2_dim=500, dec_scale_mu=0.001)
    vae_model = load_model(vae_model, args.model_path, device, copy_to_cpu=True)
    predictor = MutationEffectPredictor(data, vae_model)

    n_iter = args.n_iter
    
    df = pd.read_table(args.mutant_path)
    dscore = []
    if args.single_or_multiple == 'single':
        for m in tqdm(df.mutant):
            wt, idx, mt = m[0], int(m[1:-1]), m[-1]
            tup1 = (idx,wt,mt)
            list1 = [tup1]
            score = predictor.get_variant_delta_elbo(vae_model, list1, n_iter=n_iter)
            dscore.append(score)
    #     df.to_csv(args.save_path,sep='\t',index=False)
    # -2.03463650668
    else:
        for m in tqdm(df.mutant):
            rows = m.split(';')
            listmutant = []
            for each in rows:
                wt, idx, mt = each[0], int(each[1:-1]), each[-1]
                tup1 = (idx,wt,mt)
                listmutant.append(tup1) 
            score = predictor.get_variant_delta_elbo(vae_model, listmutant, n_iter=n_iter)
            dscore.append(score) 
    df['dscore'] = dscore
    df.to_csv(args.save_path,sep='\t',index=False)
    
    #print(predictor.get_variant_delta_elbo(vae_model, [(32, "K", "F"), (33, "D", "N")], n_iter=n_iter))
    # -16.058655309


def get_args():
    parser = argparse.ArgumentParser(description='Predict mutation effect of variants with delta elbo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--alignment_file', type=str, help='Full path to alignment file')
    parser.add_argument('-m', '--model_path', type=str, help='Full path to VAE model checkpoint')
    parser.add_argument('-tsv', '--mutant_path', type=str, help='Full path to mutant')
    parser.add_argument('-save', '--save_path', type=str, help='Full path to save')
    parser.add_argument('-compute_mode', '--single_or_multiple', type=str, help='single or multiple mutantion')

    optional = parser.add_argument_group('Optional')
    optional.add_argument('--n_iter', default=500, type=int, help='Num of iterations to calculate delta_elbo of variants,'
                                                                  'default=500')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)

import argparse

import utils

from rdkit import Chem
from rdkit.Chem import MACCSkeys

import pandas as pd


def run(infile, outfile, delimiter=',', read_header=True, id_column=0, mol_column=1, y_column=2):

    header = 0 if read_header else None

    df = pd.read_csv(infile, sep=delimiter, header=header, index_col=None, low_memory=False)
    print(df.shape)

    with open(outfile, "wt") as writer:

        for index, row in df.iterrows():
            smi = row.iloc[mol_column]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print("Failed to read mol", index, smi)
                continue
            maccs = MACCSkeys.GenMACCSKeys(mol)

            utils.write_row([row.iloc[id_column], smi, row.iloc[y_column], maccs.ToBitString()], delimiter, writer)


# Example usage:
#   python -m skl.maccs -i data/admet_group/caco2_wang/all.csv output.smi -d comma --read-header
#
def main():

    parser = argparse.ArgumentParser(description='MACCS fingerprints')
    utils.create_common_options(parser)
    parser.add_argument('-o', '--output', required=True, help="Output file (.smi, .tab, .txt)")

    args = parser.parse_args()

    delimiter = utils.read_delimiter(args.delimiter)

    run(args.input, args.output,
        read_header=args.read_header, delimiter=delimiter,
        id_column=args.id_column, mol_column=args.mol_column, y_column=args.y_column)


if __name__ == "__main__":
    main()
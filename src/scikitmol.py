import argparse

import pandas as pd
import utils
from rdkit.Chem import PandasTools
from scikit_mol.fingerprints import RDKitFingerprintTransformer
from scikit_mol.standardizer import Standardizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def run(infile, delimiter=",", read_header=True, id_column=0, mol_column=1, y_column=2):
    utils.validate_args(id_column, mol_column, y_column)

    header = 0 if read_header else None

    df = pd.read_csv(
        infile, sep=delimiter, header=header, index_col=None, low_memory=False
    )

    PandasTools.AddMoleculeColumnToFrame(df, smilesCol=df.columns[mol_column])

    print(df.shape)

    mol_list_train, mol_list_test, y_train, y_test = train_test_split(
        df.ROMol, df[df.columns[y_column]], random_state=0
    )

    print(mol_list_train.shape, mol_list_test.shape, y_train.shape, y_test.shape)

    # model = make_pipeline( make_union(
    #     RDKitFingerprintTransformer(maxPath=4),
    #     make_pipeline(MolecularDescriptorTransformer(desc_list=["MolWt", "BalabanJ"]),
    #                   PolynomialFeatures(degree=2)),
    #     n_jobs=2),
    #     Ridge(alpha=10)
    # )

    model = make_pipeline(
        Standardizer(), RDKitFingerprintTransformer(maxPath=4), Ridge(alpha=10)
    )
    model.fit(mol_list_train, y_train)

    print(f"Train score is {model.score(mol_list_train, y_train)}")
    print(f"Test score is {model.score(mol_list_test, y_test)}")


def main():
    # python -m skl.scikitmol -i data/admet_group/caco2_wang/all.csv -d comma --read-header

    parser = argparse.ArgumentParser(description="Scikit mol tests")
    parser.add_argument(
        "-i", "--infile", required=True, help="Input file (.smi, .tab, .txt)"
    )
    parser.add_argument("-d", "--delimiter", help="Delimiter when using SMILES")
    parser.add_argument(
        "--read-header",
        action="store_true",
        help="Read a header line with the field names when reading .smi or .txt",
    )
    parser.add_argument(
        "--id-column", type=int, default="0", help="Column index for molecule ID"
    )
    parser.add_argument(
        "--mol-column",
        type=int,
        default="1",
        help="Column index for molecule when using .smi",
    )
    parser.add_argument(
        "--y-column",
        type=int,
        default="2",
        help="Column index for the Y variable when using .smi",
    )

    args = parser.parse_args()

    delimiter = utils.read_delimiter(args.delimiter)

    run(
        args.infile,
        read_header=args.read_header,
        delimiter=delimiter,
        id_column=args.id_column,
        mol_column=args.mol_column,
        y_column=args.y_column,
    )


if __name__ == "__main__":
    main()

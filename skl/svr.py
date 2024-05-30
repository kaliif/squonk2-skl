import argparse

import utils

from sklearn.svm import SVR


def run(input, param_grid, scorers, delimiter=',', read_header=True, id_column=0, mol_column=1, y_column=2,
        c=[1.0], kernel=['rbf'], gamma=['scale']):

    utils.validate_args(id_column, mol_column, y_column)

    X, y, core_cols = utils.read_csv(input, read_header, delimiter, id_column, mol_column, y_column)

    utils.run_grid_search(SVR(), param_grid, scorers, X, y)


# Example usage:
#   python -m skl.svr -i data/caco2_mordred_filtered_scaled.smi -d comma --read-header --scorers R2 MAE
#
def main():

    parser = argparse.ArgumentParser(description='SVR')
    utils.create_common_options(parser)

    parser.add_argument("--scorers", nargs="+", required=True, help="Scorers")

    parser.add_argument("--c", nargs="+", type=float, default=[1.0], help="C values")
    parser.add_argument("--kernel", nargs="+", default=['rbf'], help="Kernel values")
    parser.add_argument("--gamma", nargs="+", default=['scale'], help="Gamma values")

    args = parser.parse_args()

    delimiter = utils.read_delimiter(args.delimiter)

    g = []
    for v in args.gamma:
        if v is None:
            g.append('scale')
        elif v == 'scale' or v == 'auto':
            g.append(v)
        else:
            g.append(float(v))

    param_grid = {"C": args.c, "kernel": args.kernel, "gamma": g}

    run(args.input, param_grid, args.scorers,
        read_header=args.read_header, delimiter=delimiter,
        id_column=args.id_column, mol_column=args.mol_column, y_column=args.y_column)


if __name__ == "__main__":
    main()
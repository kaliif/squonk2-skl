import argparse

import utils

from sklearn.ensemble import RandomForestRegressor


def run(input, param_grid, scorers,
        delimiter=',', read_header=True, id_column=0, mol_column=1, y_column=2, random_state=None):

    utils.validate_args(id_column, mol_column, y_column)

    X, y, core_cols = utils.read_csv(input, read_header, delimiter, id_column, mol_column, y_column)

    utils.run_grid_search(RandomForestRegressor(random_state=random_state), param_grid, scorers, X, y)


# Example usage:
#   python -m skl.rf -i data/caco2_mordred_filtered_scaled.smi -d comma --read-header --scorers R2 MAE
#
def main():

    parser = argparse.ArgumentParser(description='Random Forest')
    utils.create_common_options(parser)

    parser.add_argument("--scorers", nargs="+", required=True, help="Scorers")

    parser.add_argument("--n-estimators", nargs="+", type=int, default=[100], help="Num RF estimators")
    parser.add_argument("--min-samples-split", nargs="+", type=int, default=[2], help="Min RF samples split")
    parser.add_argument("--max-depth", nargs="+", type=int, default=[None], help="Max RF depth")
    parser.add_argument("--random-state", type=int, help="RF random state")

    args = parser.parse_args()

    delimiter = utils.read_delimiter(args.delimiter)

    param_grid = {
        "n_estimators": args.n_estimators,
        "min_samples_split": args.min_samples_split,
        "max_depth": args.max_depth}

    run(args.input, param_grid, args.scorers,
        read_header=args.read_header, delimiter=delimiter,
        id_column=args.id_column, mol_column=args.mol_column, y_column=args.y_column,
        random_state=args.random_state)


if __name__ == "__main__":
    main()
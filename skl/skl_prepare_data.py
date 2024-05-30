import argparse, time

import pandas as pd
from pandas.api.types import is_numeric_dtype

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import utils

from dm_job_utilities.dm_log import DmLog


def run(input, output,
            variance_threshold=None, remove_non_numeric=False, scale_data=False,
            delimiter="\t", id_column=1, mol_column=0, y_column=None, read_header=False, write_header=False):

    utils.validate_args(id_column, mol_column, y_column)

    header = 0 if read_header else None

    df = pd.read_csv(input, sep=delimiter, header=header, index_col=False, low_memory=False)
    DmLog.emit_event('Initial dataframe has shape {}'.format(df.shape))

    core_cols = utils.create_core_columns(df, id_column, mol_column, y_column)

    df_core = df[df.columns[[t[1] for t in core_cols]].tolist()]
    DmLog.emit_event('core shape', df_core.shape)

    cols_to_drop = [t[1] for t in core_cols]
    DmLog.emit_event('dropping cols', cols_to_drop)
    X = df.drop(df.columns[cols_to_drop], axis=1)
    DmLog.emit_event('shape after removing core columns', X.shape)

    DmLog.emit_event('Considering shape {}'.format(X.shape))
    removed_count = 0
    if remove_non_numeric:
        DmLog.emit_event('removing non-numeric columns')
        for k in X.keys():
            if not is_numeric_dtype(X[k]):
                removed_count += 1
                del X[k]
        DmLog.emit_event('removed {} columns. shape is now {}'.format(removed_count, X.shape))
    names = X.columns.tolist()

    if variance_threshold is not None:
        threshold = variance_threshold * (1 - variance_threshold)
        DmLog.emit_event('reducing dimensionality using threshold of', threshold)
        selector = VarianceThreshold(threshold=threshold)
        reduced = selector.fit_transform(X)
        names = selector.get_feature_names_out()
        X = pd.DataFrame(data=reduced, columns=names, index=df.index)
        DmLog.emit_event('shape reduced to {}'.format(X.shape))

    if scale_data:
        DmLog.emit_event('scaling data')
        ss = StandardScaler()
        scaled = ss.fit_transform(X)
        X = pd.DataFrame(data=scaled, columns=names, index=df.index)
        DmLog.emit_event('scaled shape {}'.format(X.shape))

    result = pd.concat([df_core, X], axis=1, join="inner")
    DmLog.emit_event('recombined shape:', result.shape)

    if output:
        with open(output, 'wt') as out:

            if write_header:
                out.write(delimiter.join([t[2] for t in core_cols]))
                for name in names:
                    out.write(delimiter + name)
                out.write('\n')

            for i in range(result.shape[0]):

                # add the core col values
                row = []
                for c in range(len(core_cols)):
                    row.append(result.iloc[i, c])
                # add the descriptors
                row.extend(result.iloc[i, len(core_cols):])

                utils.write_row(row, delimiter, out)

            DmLog.emit_event('written output to', output)

    return result


def main():

    # Example:
    #   python -m skl.skl_prepare_data -i data/descriptors2d_10000.smi -o foo.smi -d tab --id-column 0 --mol-column 1 --y-column 2 \
    #     --read-header --write-header --remove-non-numeric -t 0.9 --scale-data

    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('-i', '--input', required=True, help="Input file (.smi, .tab, .txt)")
    parser.add_argument('-o', '--output', help="Output file (.smi)")

    parser.add_argument('-t', '--variance-threshold', type=float, help="Variance threshold")
    parser.add_argument('--remove-non-numeric', action='store_true',
                        help="Remove columns that are not numeric")
    parser.add_argument('--scale-data', action='store_true',
                        help="Scale the values for each column to have mean of 0 and sd of 1")

    # to pass tab as the delimiter specify it as $'\t' or use one of the symbolic names 'comma', 'tab', 'space' or 'pipe'
    parser.add_argument('-d', '--delimiter', help="Delimiter when using SMILES")
    parser.add_argument('--id-column', type=int, help="Column index for molecule ID")
    parser.add_argument('--mol-column', type=int, help="Column index for molecule when using .smi")
    parser.add_argument('--y-column', type=int, help="Column index for the Y variable when using .smi")
    parser.add_argument('--read-header', action='store_true',
                        help="Read a header line with the field names when reading .smi or .txt")
    parser.add_argument('--write-header', action='store_true', help='Write a header line when writing .smi or .txt')

    args = parser.parse_args()
    DmLog.emit_event("skl_prepare_data: ", args)

    delimiter = utils.read_delimiter(args.delimiter)

    run(args.input, args.output,
        variance_threshold=args.variance_threshold, remove_non_numeric=args.remove_non_numeric,
        scale_data=args.scale_data, delimiter=delimiter,
        id_column=args.id_column, mol_column=args.mol_column, y_column=args.y_column,
        read_header=args.read_header, write_header=args.write_header)


if __name__ == "__main__":
    main()

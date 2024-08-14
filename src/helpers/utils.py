import argparse
import os
import sys
import time
from collections import OrderedDict
from math import floor, log10

import pandas as pd
from dm_job_utilities.dm_log import DmLog
from jaqpotpy.datasets import SmilesDataset
from jaqpotpy.descriptors.molecular import (
    MACCSKeysFingerprint,
    MordredDescriptors,
    RDKitDescriptors,
    TopologicalFingerprint,
)
from jaqpotpy.doa.doa import Leverage
from jaqpotpy.models import MolecularSKLearn
from jaqpotpy.models.evaluator import Evaluator
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    make_scorer,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from tdc.benchmark_group import admet_group

from helpers.cross_train import cross_train_sklearn
from helpers.get_data import get_dataset

default_num_chars = 2
default_num_levels = 2

delimiter_choices = {
    "space": None,
    "tab": "\t",
    "comma": ",",
    "pipe": "|",
}


def create_common_args():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--data", required=True, help="Training data")
    argParser.add_argument(
        "-f",
        "--featurizers",
        nargs="+",
        choices=["mordred", "maccs", "topo", "rdkit"],
        help="Molecular feature generators to use",
    )
    argParser.add_argument(
        "-s",
        "--scoring-functions",
        nargs="+",
        choices=["MAE", "ACC", "AUC", "AUPRC", "SPM"],
        help="Scoring functions to use in the Evaluator",
    )
    argParser.add_argument("--doa", help="DOA algorithm (leverage or None)")
    return argParser


def create_featurizer(name: str):
    if name == "mordred":
        return MordredDescriptors()
    elif name == "maccs":
        return MACCSKeysFingerprint()
    elif name == "topo":
        return TopologicalFingerprint()
    elif name == "rdkit":
        return RDKitDescriptors()
    else:
        raise ValueError("Invalid featurizer name: " + name)


def create_featurizers(names):
    return [create_featurizer(n) for n in names]


def create_evaluator(scoring_function_names):
    val = Evaluator()
    for name in scoring_function_names:
        if name == "MAE":
            val.register_scoring_function(name, mean_absolute_error)
        elif name == "ACC":
            val.register_scoring_function(name, accuracy_score)
        elif name == "AUC":
            val.register_scoring_function(name, roc_auc_score)
        elif name == "AUPRC":
            val.register_scoring_function(name, average_precision_score)
        elif name == "SPM":
            val.register_scoring_function(name, spearmanr)
        else:
            raise ValueError("Invalid scoring function name: " + name)
    return val


def create_doa(s_doa):
    """
    Domain of applicability will be Leverage() or None
    :param s_doa:
    :return:
    """
    if s_doa is None:
        return None
    elif s_doa.lower() == "leverage":
        return Leverage()
    else:
        print(
            "invalid value for doa. only leverage is supported. {} was specified".format(
                s_doa
            )
        )
        exit(1)


class Runner:
    def __init__(
        self, dataset_name: str, models: dict, doa, evaluator, featurizers, task
    ):
        self.group = admet_group(path="data/")
        self.benchmark, self.name = get_dataset(dataset_name, self.group)

        self.train_val = self.benchmark["train_val"]
        self.test = self.benchmark["test"]

        self.models = models
        self.doa = doa
        self.evaluator = evaluator
        self.featurizers = featurizers
        self.task = task

    def run_cross_validation(self):
        print("Evaluating {} models".format(len(self.models)))

        t0 = time.time()
        results = []
        for featurizer in self.featurizers:
            dummy_train = SmilesDataset(
                smiles=self.train_val["Drug"],
                y=self.train_val["Y"],
                featurizer=featurizer,
                task=self.task,
            )
            for key in self.models:
                model = self.models[key]
                skl_model = MolecularSKLearn(
                    dummy_train, doa=self.doa, model=model, eval=self.evaluator
                )

                # Cross Validate and check robustness
                evaluation = cross_train_sklearn(
                    self.group, skl_model, self.name, self.test, task=self.task
                )
                results.append((key + ", " + str(featurizer), evaluation))
        t1 = time.time()

        print("\n\n")
        for result in results:
            print("Evaluation of the model:", result[0], result[1])
        print("Execution took {} seconds".format(round(t1 - t0)))
        return results


def log(*args, **kwargs):
    """Log output to STDERR"""
    print(*args, file=sys.stderr, **kwargs)


def get_path_from_digest(
    digest, num_chars=default_num_chars, num_levels=default_num_levels
):
    parts = []
    start = 0
    for _ in range(0, num_levels):
        end = start + num_chars
        p = digest[start:end]
        parts.append(p)
        start = start + num_chars
    return parts


def expand_path(path):
    """
    Create any necessary directories to ensure that the file path is valid

    :param path: a filename or directory that might or not exist
    """
    head_tail = os.path.split(path)
    if head_tail[0]:
        if not os.path.isdir(head_tail[0]):
            log("Creating directories for", head_tail[0])
            os.makedirs(head_tail[0], exist_ok=True)


def UpdateChargeFlagInAtomBlock(mb):
    """
    See https://sourceforge.net/p/rdkit/mailman/message/36425493/
    """
    f = "{:>10s}" * 3 + "{:>2}{:>4s}" + "{:>3s}" * 11
    chgs = []  # list of charges
    lines = mb.split("\n")
    if mb[0] == "" or mb[0] == "\n":
        del lines[0]
    CTAB = lines[2]
    atomCount = int(CTAB.split()[0])
    # parse mb line per line
    for l in lines:
        # look for M CHG property
        if l[0:6] == "M  CHG":
            records = l.split()[
                3:
            ]  # M  CHG X is not needed for parsing, the info we want comes afterwards
            # record each charge into a list
            for i in range(0, len(records), 2):
                idx = records[i]
                chg = records[i + 1]
                chgs.append((int(idx), int(chg)))  # sort tuples by first element?
            break  # stop iterating

    # sort by idx in order to parse the molblock only once more
    chgs = sorted(chgs, key=lambda x: x[0])

    # that we have a list for the current molblock, attribute each charges
    for chg in chgs:
        i = 3
        while (
            i < 3 + atomCount
        ):  # do not read from beginning each time, rather continue parsing mb!
            # when finding the idx of the atom we want to update, extract all fields and rewrite whole sequence
            if (
                i - 2 == chg[0]
            ):  # -4 to take into account the CTAB headers, +1 because idx begin at 1 and not 0
                fields = lines[i].split()
                x = fields[0]
                y = fields[1]
                z = fields[2]
                symb = fields[3]
                massDiff = fields[4]
                charge = fields[5]
                sp = fields[6]
                hc = fields[7]
                scb = fields[8]
                v = fields[9]
                hd = fields[10]
                nu1 = fields[11]
                nu2 = fields[12]
                aamn = fields[13]
                irf = fields[14]
                ecf = fields[15]
                # update charge flag
                if chg[1] == -1:
                    charge = "5"
                elif chg[1] == -2:
                    charge = "6"
                elif chg[1] == -3:
                    charge = "7"
                elif chg[1] == 1:
                    charge = "3"
                elif chg[1] == 2:
                    charge = "2"
                elif chg[1] == 3:
                    charge = "1"
                else:
                    print(
                        "ERROR! "
                        + str(lines[0])
                        + "unknown charge flag: "
                        + str(chg[1])
                    )  # print name then go to next chg
                    break
                # update modatom block line
                lines[i] = f.format(
                    x,
                    y,
                    z,
                    symb,
                    massDiff,
                    charge,
                    sp,
                    hc,
                    scb,
                    v,
                    hd,
                    nu1,
                    nu2,
                    aamn,
                    irf,
                    ecf,
                )
            i += 1
    # print("\n".join(lines))
    del lines[-1]  # remove empty element left because last character before $$$$ is \n
    upmb = "\n" + "\n".join(lines)
    return upmb


def read_delimiter(delimiter):
    return delimiter_choices.get(delimiter, delimiter)


def calc_geometric_mean(scores):
    total = 1.0
    for score in scores:
        total = total * score
    result = total ** (1.0 / len(scores))
    return result


def round_to_significant_number(val, sig):
    """
    Round the value to the specified number of significant numbers
    :param val: The number to round
    :param sig: Number of significant numbers
    :return:
    """
    return round(val, sig - int(floor(log10(abs(val)))) - 1)


def is_type(value, typ):
    if value is not None:
        try:
            i = typ(value)
            return 1, i
        except:
            return -1, value
    else:
        return 0, value


def validate_args(id_column, mol_column, y_column):
    if id_column == mol_column:
        DmLog.emit_event("ERROR: mol_column and id_column must be different")
        exit(1)
    if y_column == id_column:
        DmLog.emit_event("ERROR: y_column and id_column must be different")
        exit(1)
    if y_column == mol_column:
        DmLog.emit_event("ERROR: y_column and mol_column must be different")
        exit(1)


def create_core_columns(df, id_column, mol_column, y_column):
    core_cols = [
        ("SMILES", mol_column, df.columns.to_list()[mol_column]),
        ("ID", id_column, df.columns.to_list()[id_column]),
    ]

    if y_column is not None:
        core_cols.append(("Y", y_column, df.columns[y_column]))

    core_cols.sort(key=lambda x: x[1])
    return core_cols


def write_row(row, delimiter, out):
    count = 0
    for item in row:
        if count > 0:
            out.write(delimiter)

        if type(item) == str and delimiter in item:
            out.write('"' + item + '"')
        else:
            out.write(str(item))

        count += 1
    out.write("\n")


def create_scorers(scoring_function_names):
    scorers = OrderedDict()
    for name in scoring_function_names:
        if name == "MAE":
            scorers[name] = make_scorer(mean_absolute_error)
        elif name == "ACC":
            scorers[name] = make_scorer(accuracy_score)
        elif name == "AUC":
            scorers[name] = make_scorer(roc_auc_score)
        elif name == "AUPRC":
            scorers[name] = make_scorer(average_precision_score)
        elif name == "SPM":
            scorers[name] = make_scorer(spearmanr)
        elif name == "R2":
            scorers[name] = make_scorer(r2_score)
        else:
            raise ValueError("Unsupported scoring function name: " + name)
    return scorers


def create_common_options(parser):
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
    parser.add_argument(
        "--fp-column",
        type=int,
        help="Column index for a bit string that contains the fingerprint"
        + " (if not provided, then all other columns are assumed to have the data items)",
    )


def read_csv(
    filename: str,
    read_header: bool,
    delimiter: str,
    id_column: int,
    mol_column: int,
    y_column: int,
):
    header = 0 if read_header else None

    df = pd.read_csv(
        filename, sep=delimiter, header=header, index_col=None, low_memory=False
    )

    core_cols = create_core_columns(df, id_column, mol_column, y_column)
    # print(core_cols)

    y = df.iloc[:, y_column]
    # print(y.shape)
    df.drop(df.columns[[id_column, mol_column, y_column]], axis=1, inplace=True)
    X = df

    return X, y, core_cols


def run_grid_search(method, param_grid, scorers, X, y):
    scoring = create_scorers(scorers)

    gs = GridSearchCV(
        method,
        param_grid=param_grid,
        scoring=scoring,
        refit=scorers[0],
        n_jobs=-1,
        return_train_score=True,
        # verbose=2
    )

    gs.fit(X, y)
    results = gs.cv_results_
    best_estimator = gs.best_estimator_
    best_params = best_estimator.get_params()
    print("Best model")
    for k in param_grid:
        print(k, best_params[k])
    print("best_score:", gs.best_score_)
    print("best_index:", gs.best_index_)
    for scorer in scorers:
        print(scorer, results["mean_test_" + scorer][gs.best_index_])

    return best_estimator

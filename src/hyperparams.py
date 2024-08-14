import argparse
import json
import logging

# import os
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV

from arrayinput import (
    BaseArrayType,
    BoolArray,
    ChoiceArray,
    FloatArray,
    IntArray,
    NumberArray,
    UniqueArray,
)
from helpers.utils import read_csv, read_delimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_choices = {
    "RandomForestRegressor": RandomForestRegressor,
}

search_choices = {
    "RandomForestRegressor": RandomForestRegressor,
}


warnings.filterwarnings("error", category=FitFailedWarning)


def get_model(model_class):
    try:
        return model_choices[model_class]
    except KeyError as exc:
        raise ValueError(f"Unsupported model type: {model_class}") from exc


class CriterionArray(ChoiceArray):
    err_msg = "{} is not a valid comma-separated list of criterions."
    # cant't use name 'choices' here, this is defined in base class
    # and I should define it in argparse itself.. which then would
    # create a conflict
    options = ("squared_error", "absolute_error", "friedman_mse", "poisson")


class MaxFeaturesArray(BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of max_features."
    options = ("sqrt", "log2")

    def get_functions(self):
        return (self.parse_none, self.parse_choices, self.parse_int, self.parse_float)


# what about uniqueness?
class NoneOrIntArray(UniqueArray, BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of max_depth."
    functions = (
        BaseArrayType.parse_none,
        BaseArrayType.parse_int,
    )


class NoneOrNumberArray(UniqueArray, BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of max_depth."
    functions = (
        BaseArrayType.parse_none,
        BaseArrayType.parse_int,
        BaseArrayType.parse_float,
    )


# the following do not belong to RandomForestRegressor but
# GridSearchCV. using them the same way for now, but potentially
# the're going to need different handling and better validation
class ScoringArray(BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of scoring."
    options = (
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_root_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
        "neg_mean_absolute_percentage_error",
        "d2_absolute_error_score",
    )

    def get_functions(self):
        return (self.parse_none, self.parse_choices)


# won't be used as an array, shoehorning it in right now
class PreDispatchArray(UniqueArray, BaseArrayType):
    err_msg = "{} is not a valid value for pre_dispatch."
    functions = (
        BaseArrayType.parse_none,
        BaseArrayType.parse_int,
        BaseArrayType.parse_str,
    )


class ErrorScoreArray(BaseArrayType):
    err_msg = "{} is not a valid value for error_score."
    options = ("raise",)

    def get_functions(self):
        return (self.parse_choices, self.parse_int, self.parse_float)


def select_results(results, n_top=3) -> list[dict]:
    result = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            d = {
                "rank": i,
                "mean_test_score": results["mean_test_score"][candidate],
                "std_test_score": results["std_test_score"][candidate],
            }

            try:
                d["mean_train_score"] = results["mean_train_score"][candidate]
                d["std_train_score"] = results["std_train_score"][candidate]
            except KeyError:
                pass

            d["params"] = results["params"][candidate]

            result.append(d)
    return result


def run(
    infile,
    outfile,
    delimiter=",",
    id_column=1,
    mol_column=0,
    y_column=2,
    read_header=False,
    n_top=5,
    hyper_params=None,
    search_params=None,
) -> None:
    # current_path = Path(__file__)

    model_class = "RandomForestRegressor"

    if hyper_params is None:
        hyper_params = {}

    if search_params is None:
        search_params = {}

    # it looks like read_csv is able to process smiles file just the same
    X, y, _ = read_csv(
        filename=infile,
        read_header=read_header,
        delimiter=delimiter,
        id_column=id_column,
        mol_column=mol_column,
        y_column=y_column,
    )

    # fewer descriptors for testing
    X = X.iloc[:, :40]

    model = get_model(model_class)()

    param_grid = {}
    for k in model.get_params().keys():
        param = hyper_params.get(k, None)
        # if missing, probably some non-tunable param, like verbosity
        if param:
            param_grid[k] = param
            logger.info("%s param %s: %s", model_class, k, param)

    search = GridSearchCV(model, param_grid=param_grid, **search_params)
    t0 = time.time()

    result = None
    try:
        result = search.fit(X, y)
    except FitFailedWarning as exc:
        # when parameter search encounters invalid value amongst valid
        # ones, it throws a warning and keeps going. Catch the warning
        # and handle
        logger.error(exc)
        for k in exc.args[0].split("\n"):
            if k.find("InvalidParameterError:") > 0:
                logger.error(k.split("InvalidParameterError:")[1].strip())
    except ValueError as exc:
        # when all parameters are invalid however, ValueError is thrown
        logger.error(exc)
        for k in exc.args[0].split("\n"):
            if k.find("InvalidParameterError:") > 0:
                logger.error(k.split("InvalidParameterError:")[1].strip())

    t1 = time.time()

    logger.info("Runtime: %.2f", t1 - t0)
    if result:
        selected = select_results(result.cv_results_, n_top=n_top)
        filename = outfile
        if not filename.endswith(".json"):
            filename = filename + ".json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=4)
    else:
        logger.info("No results to report")


def main():
    # Example:
    # python src/hyperparams.py
    #     --read-header\
    #     --outfile=outfile.json\
    #     --infile=data/caco2_mordred_filtered_scaled.smi\
    #     --bootstrap=True,False\
    #     --ccp_alpha=0.0,0.1,0.5

    parser = argparse.ArgumentParser(description="Prepare data")

    parser.add_argument("--infile", required=True, help="Input training set file")
    parser.add_argument("-o", "--outfile", help="Output file (.json)")

    parser.add_argument(
        "-d", "--delimiter", default="comma", help="Delimiter when using SMILES"
    )
    parser.add_argument(
        "--id-column", type=int, default=1, help="Column index for molecule ID"
    )
    parser.add_argument(
        "--mol-column",
        type=int,
        default=0,
        help="Column index for molecule when using .smi",
    )
    parser.add_argument(
        "--y-column",
        type=int,
        default=2,
        help="Column index for the Y variable when using .smi",
    )
    parser.add_argument(
        "--n_top_results",
        type=int,
        default=5,
        help="Report top n results in output file",
    )
    parser.add_argument(
        "--read-header",
        action="store_true",
        help="Read a header line with the field names when reading .smi or .txt",
    )

    # RFR params
    parser.add_argument(
        "--bootstrap",
        action=BoolArray,
        default=[True],
        help=(
            "Whether bootstrap samples are used when building trees."
            + " If False, the whole dataset is used to build each tree."
        ),
    )
    parser.add_argument(
        "--ccp_alpha",
        action=FloatArray,
        default=[0.0],
        help=(
            "Complexity parameter used for Minimal Cost-Complexity Pruning."
            + " The subtree with the largest cost complexity that is smaller"
            + " than ccp_alpha will be chosen. By default, no pruning is performed."
        ),
    )
    parser.add_argument(
        "-c",
        "--criterion",
        default=["squared_error"],
        action=CriterionArray,
        help=(
            "The function to measure the quality of a split. Supported"
            + ' criteria are "squared_error" for the mean squared error, which'
            + " is equal to variance reduction as feature selection criterion"
            + " and minimizes the L2 loss using the mean of each terminal node,"
            + ' "friedman_mse", which uses mean squared error with Friedman’s'
            + ' improvement score for potential splits, "absolute_error" for the'
            + " mean absolute error, which minimizes the L1 loss using the"
            + ' median of each terminal node, and "poisson" which uses reduction'
            + " in Poisson deviance to find splits. Training using"
            + ' "absolute_error" is significantly slower than when using'
            + ' "squared_error".'
        ),
    )
    parser.add_argument(
        "--max_features",
        default=[1.0],
        action=MaxFeaturesArray,
        help=(
            "The number of features to consider when looking for the best split:"
            + " If int, then consider max_features features at each split."
            + " If float, then max_features is a fraction and"
            + " max(1, int(max_features * n_features_in_)) features are"
            + " considered at each split."
            + ' If "sqrt", then max_features=sqrt(n_features).'
            + ' If "log2", then max_features=log2(n_features).'
            + " If None or 1.0, then max_features=n_features."
        ),
    )
    parser.add_argument(
        "--max_depth",
        default=[None],
        action=NoneOrIntArray,
        help=(
            "The maximum depth of the tree. If None, then nodes are expanded"
            + " until all leaves are pure or until all leaves contain less"
            + " than min_samples_split samples."
        ),
    )
    parser.add_argument(
        "--max_leaf_nodes",
        default=[None],
        action=NoneOrIntArray,
        help=(
            "Grow trees with max_leaf_nodes in best-first fashion. Best nodes"
            + " are defined as relative reduction in impurity. If None then"
            + " unlimited number of leaf nodes."
        ),
    )
    parser.add_argument(
        "--max_samples",
        default=[None],
        action=NoneOrNumberArray,
        help=(
            "If bootstrap is True, the number of samples to draw from"
            + " X to train each base estimator."
            + " If None (default), then draw X.shape[0] samples."
            + " If int, then draw max_samples samples."
            + " If float, then draw max(round(n_samples * max_samples), 1)"
            + " samples. Thus, max_samples should be in the interval (0.0, 1.0]."
        ),
    )
    parser.add_argument(
        "--min_impurity_decrease",
        default=[0.0],
        action=FloatArray,
        help=(
            "A node will be split if this split induces a decrease of the"
            + " impurity greater than or equal to this value. The weighted"
            + " impurity decrease equation is the following: N_t / N * (impurity"
            + " - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity)"
            + " where N is the total number of samples, N_t is the number of"
            + " samples at the current node, N_t_L is the number of samples in"
            + " the left child, and N_t_R is the number of samples in the right"
            + " child. N, N_t, N_t_R and N_t_L all refer to the weighted sum, if"
            + " sample_weight is passed."
        ),
    )
    parser.add_argument(
        "--min_samples_leaf",
        default=[1],
        action=NumberArray,
        help=(
            "The minimum number of samples required to be at a leaf node. A"
            + " split point at any depth will only be considered if it leaves at"
            + " least min_samples_leaf training samples in each of the left and"
            + " right branches. This may have the effect of smoothing the model,"
            + " especially in regression. If int, then consider min_samples_leaf"
            + " as the minimum number. If float, then min_samples_leaf is a"
            + " fraction and ceil(min_samples_leaf * n_samples) are the minimum"
            + " number of samples for each node."
        ),
    )
    parser.add_argument(
        "--min_samples_split",
        default=[2],
        action=NumberArray,
        help=(
            "The minimum number of samples required to split an internal node:"
            + " If int, then consider min_samples_split as the minimum number."
            + " If float, then min_samples_split is a fraction and"
            + " ceil(min_samples_split * n_samples) are the minimum number"
            + " of samples for each split."
        ),
    )
    parser.add_argument(
        "--min_weight_fraction_leaf",
        default=[0.0],
        action=FloatArray,
        help=(
            "The minimum weighted fraction of the sum total of weights (of all"
            + " the input samples) required to be at a leaf node. Samples have"
            + " equal weight when sample_weight is not provided."
        ),
    )
    parser.add_argument(
        "--n_estimators",
        default=[100],
        action=IntArray,
        help="The number of trees in the forest.",
    )
    parser.add_argument(
        "--n_jobs_estimator",
        default=None,
        type=int,
        help=(
            "The number of jobs to run in parallel. fit, predict,"
            + " decision_path and apply are all parallelized over the trees."
            + " None means 1 unless in a joblib.parallel_backend context. -1"
            + " means using all processors."
        ),
    )
    # NB! this can also be callable, this option is not implemented here
    parser.add_argument(
        "--oob_score",
        default=[False],
        action=BoolArray,
        help=(
            "Whether to use out-of-bag samples to estimate the generalization"
            + " score. By default, r2_score is used."
        ),
    )
    # parser.add_argument(
    #     '--random_state',
    #     default=None,
    #     help=('Controls both the randomness of the bootstrapping of the samples'
    #           + ' used when building trees (if bootstrap=True) and the sampling of'
    #           + ' the features to consider when looking for the best split at each'
    #           + ' node (if max_features < n_features).'),
    # )
    parser.add_argument(
        "--verbose_estimator",
        default=0,
        type=int,
        help="Controls the verbosity when fitting and predicting.",
    )
    parser.add_argument(
        "--warm_start",
        default=[False],
        action=BoolArray,
        help=(
            "When set to True, reuse the solution of the previous call to fit"
            + " and add more estimators to the ensemble, otherwise, just fit a"
            + " whole new forest. See Glossary and Fitting additional trees for"
            + " details."
        ),
    )
    # the n_features this depends on is features in input data. cannot
    # validate this here
    parser.add_argument(
        "--monotonic_cst",
        default=[None],
        action=NoneOrIntArray,
        help=(
            "Indicates the monotonicity constraint to enforce on each feature."
            + " 1: monotonically increasing"
            + " 0: no constraint"
            + " -1: monotonically decreasing"
            + " If monotonic_cst is None, no constraints are applied."
            + " Monotonicity constraints are not supported for:"
            + " multioutput regressions (i.e. when n_outputs_ > 1),"
            + " regressions trained on data with missing values."
        ),
    )
    # not used. will it be used at all?
    parser.add_argument(
        "--search_algorithm",
        choices=[
            "GridSearchCV",
            "HalvingGridSearchCV",
            "RandomizedSearchCV",
            "HalvingRandomSearchCV",
        ],
        default=["GridSearchCV"],
        help=("Algorithm to be used in parameter tuning"),
    )

    # GridSearchCV params

    # this is a curious parameter. It's not one of estimator's that's
    # being tuned, but belongs to search algorithm itself. In the
    # spirit of scikit, it allows multiple types of parameters so
    # using the array action seems conveninent.
    # NB! it has several other options, dicts and callables that are note
    # currently implemented
    parser.add_argument(
        "--scoring",
        default=[None],
        action=ScoringArray,
        help=(
            "Strategy to evaluate the performance of the cross-validated model"
            + " on the test set."
            + " If scoring represents a single score, one can use:"
            + " a single string (see The scoring parameter: defining model"
            + " evaluation rules)"
            + " If scoring represents multiple scores, one can use:"
            + " a list or tuple of unique strings."
            + " See Specifying multiple metrics for evaluation for an example."
        ),
    )
    parser.add_argument(
        "--n_jobs_search",
        default=None,
        type=int,
        help=(
            "The number of jobs to run in parallel."
            + " None means 1 unless in a joblib.parallel_backend context. -1"
            + " means using all processors."
        ),
    )
    # NB! more complex options, object and lists not implemented

    parser.add_argument(
        "--cv",
        default=None,
        type=int,
        help=(
            "Determines the cross-validation splitting strategy."
            + " Possible inputs for cv are:"
            + " None, to use the default 5-fold cross validation"
            + " integer, to specify the number of folds in a (Stratified)KFold"
        ),
    )
    parser.add_argument(
        "--verbose_search",
        default=0,
        type=int,
        help="Controls the verbosity: the higher, the more messages.",
    )
    parser.add_argument(
        "--pre_dispatch",
        default=["2*n_jobs"],
        action=PreDispatchArray,
        help=(
            "Controls the number of jobs that get dispatched during parallel"
            + " execution. Reducing this number can be useful to avoid an explosion"
            + " of memory consumption when more jobs get dispatched than CPUs can"
            + " process. This parameter can be:"
            + " None, in which case all the jobs are immediately created and spawned."
            + " Use this for lightweight and fast-running jobs, to avoid delays due"
            + " to on-demand spawning of the jobs"
            + " An int, giving the exact number of total jobs that are spawned"
            + " A str, giving an expression as a function of n_jobs, as in '2*n_jobs'"
        ),
    )
    parser.add_argument(
        "--error_score",
        default=[np.nan],
        action=ErrorScoreArray,
        help=(
            "Value to assign to the score if an error occurs in estimator fitting."
            + " If set to 'raise', the error is raised."
            + " If a numeric value is given, FitFailedWarning is raised.  This parameter"
            + " does not affect the refit step, which will always raise the error."
        ),
    )
    parser.add_argument(
        "--return_train_score",
        default=False,
        action="store_true",
        help=(
            "If False, the cv_results_ attribute will not include training scores."
            + " Computing training scores is used to get insights on how different"
            + " parameter settings impact the overfitting/underfitting trade-off."
            + " However computing the scores on the training set can be"
            + " computationally expensive and is not strictly required to select"
            + " the parameters that yield the best generalization performance."
        ),
    )

    args = parser.parse_args()

    hyper_params = {
        "bootstrap": args.bootstrap,
        "ccp_alpha": args.ccp_alpha,
        "criterion": args.criterion,
        "max_features": args.max_features,
        "max_depth": args.max_depth,
        "max_leaf_nodes": args.max_leaf_nodes,
        "max_samples": args.max_samples,
        "min_impurity_decrease": args.min_impurity_decrease,
        "min_samples_leaf": args.min_samples_leaf,
        "min_samples_split": args.min_samples_split,
        "min_weight_fraction_leaf": args.min_weight_fraction_leaf,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs_estimator,
        "oob_score": args.oob_score,
        "verbose": args.verbose_estimator,
        "warm_start": args.warm_start,
        "monotonic_cst": args.monotonic_cst,
    }

    search_params = {
        "scoring": args.scoring,
        "n_jobs": args.n_jobs_search,
        # setting this explicitly false for now. Only parameter values
        # are returned, not the fitted model
        "refit": False,
        "cv": args.cv,
        "verbose": args.verbose_search,
        # another unorthodox use of arrayinput, always one element, just need the parsing
        "pre_dispatch": args.pre_dispatch[0],
        "error_score": args.error_score[0],
        "return_train_score": args.return_train_score,
    }

    # this is not a tunable parameter
    if len(search_params["scoring"]) == 1:
        search_params["scoring"] = search_params["scoring"][0]

    delimiter = read_delimiter(args.delimiter)

    run(
        args.infile,
        args.outfile,
        delimiter=delimiter,
        n_top=args.n_top_results,
        id_column=args.id_column,
        mol_column=args.mol_column,
        y_column=args.y_column,
        read_header=args.read_header,
        hyper_params=hyper_params,
        search_params=search_params,
    )


if __name__ == "__main__":
    main()

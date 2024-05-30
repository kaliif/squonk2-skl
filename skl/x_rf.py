
from sklearn.ensemble import RandomForestRegressor

from skl.helpers.utils import create_common_args
from skl.helpers.utils import create_featurizers
from skl.helpers.utils import create_evaluator
from skl.helpers.utils import create_doa
from skl.helpers.utils import Runner


def run(data, task, doa_alg, n_estimators, min_samples_split, max_depth, featurizers, scoring_functions, random_state=None):

    doa = create_doa(doa_alg)
    # Declare the different variants of the model's algorithm
    models = {}
    for e in n_estimators:
        for s in min_samples_split:
            for d in max_depth:
                model = RandomForestRegressor(n_estimators=e,
                                              min_samples_split=s,
                                              max_depth=d,
                                              random_state=random_state)
                key = "n_estimators={}, min_samples_split={}, max_depth={}".format(str(e), str(s), str(d))
                models[key] = model
                print("added model", key)

    # create the Featurizer and the Evaluator's metrics
    featurizers = create_featurizers(featurizers)
    val = create_evaluator(scoring_functions)

    runner = Runner(data, models, doa, val, featurizers, task)
    results = runner.run_cross_validation()


# Example usage:
#   python -m skl.x_rf -d Caco2_Wang -f mordred --max-depth 9 -s MAE --task regression
def main():

    argParser = create_common_args()
    argParser.add_argument("--n-estimators", nargs="+", type=int, default=[200], help="Num RF estimators")
    argParser.add_argument("--min-samples-split", nargs="+", type=int, default=[5], help="Min RF samples split")
    argParser.add_argument("--max-depth", nargs="+", type=int, default=[9], help="Max RF depth")
    argParser.add_argument("--random-state", type=int, default=8, help="RF random state")
    argParser.add_argument("-t", "--task",
                           required=True,
                           choices=["regression", "classification"],
                           help="regression or classification model")

    args = argParser.parse_args()

    run(args.data, args.task, args.doa, args.n_estimators, args.min_samples_split,
        args.max_depth, args.featurizers, args.scoring_functions, random_state=args.random_state)


if __name__ == "__main__":
    main()

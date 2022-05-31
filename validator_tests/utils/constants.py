JOBIDS_FILENAME = "all_validator_jobids.json"
VALIDATOR_TESTS_FOLDER = "validator_tests"
ALL_DFS_FILENAME = "all_dfs.pkl"
PER_SRC_FILENAME = "per_src_threshold.pkl"
PER_SRC_PER_ADAPTER_FILENAME = "per_src_threshold_per_adapter.pkl"
PROCESSED_DF_FILENAME = "all_dfs_processed.pkl"
TARGET_ACCURACY = "target_train_micro"
TARGET_VAL_ACCURACY = "target_val_micro"
NUM_ADAPTERS = 10
EXPECTED_NUMBER_OF_CHECKPOINTS = NUM_ADAPTERS * 100 * 20


def exp_group_args():
    return [
        "exp_groups",
        "exp_group_prefix",
        "exp_group_suffix",
        "exp_group_includes",
        "exp_group_excludes",
    ]


def add_exp_group_args(parser):
    parser.add_argument("--exp_groups", nargs="+", type=str, default=[])
    x = exp_group_args()
    x.remove("exp_groups")
    for k in x:
        parser.add_argument(f"--{k}", type=str)

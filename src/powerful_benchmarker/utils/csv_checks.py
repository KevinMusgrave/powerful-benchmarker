import json
import os

import numpy as np
import pandas as pd


def folder_is_trial(x):
    basename = os.path.basename(x)
    return basename.isdigit()


def get_validator_filename(validator):
    if validator == "entropy_diversity":
        return "validator_MultipleValidators"
    if validator in ["DEV", "DEV_binary"]:
        return "validator_DeepEmbeddedValidator"
    if validator in ["KNN", "SND"]:
        return f"validator_{validator}Validator"
    if validator == "oracle":
        return f"validator_AccuracyValidator"
    return f"validator_{validator}"


def get_commandline_config(exp_path, exp_num=None):
    if exp_num is not None:
        exp_path = os.path.join(exp_path, str(exp_num))
    config = os.path.join(exp_path, "configs", "commandline_args.json")
    with open(config, "r") as f:
        config = json.load(f)
    return config


def filter_by_trial(df, exp_num):
    return df[df["trial"] == exp_num]


def trials_df_is_not_empty(trials_df):
    return trials_df is not None and len(trials_df) > 0


def assert_score_history(score_dict):
    assert score_dict["epochs"][0] == 0
    # skip 0th epoch which is the accuracy of the pretrained model
    for k in ["score_history", "raw_score_history", "epochs"]:
        score_dict[k] = score_dict[k][1:]
    assert (
        len(score_dict["score_history"])
        == len(score_dict["raw_score_history"])
        == len(score_dict["epochs"])
    )
    # none of the experiments use a score normalizer, so this should be true
    assert np.array_equal(
        score_dict["score_history"], score_dict["raw_score_history"], equal_nan=True
    )


def drop_bad_rows_from_scores_csv(trials_df, scores_df, raise_exception):
    bad_index = []
    for index, row in trials_df.iterrows():
        curr_value = row["value"]
        curr_exp_num = row["number"]

        exp_data = filter_by_trial(scores_df, curr_exp_num)
        if exp_data.shape[0] == 0:
            continue
        for index2, row2 in exp_data.iterrows():
            if row2["trial"] != curr_exp_num:
                raise ValueError
            if not np.isclose(row2["validation_score"], curr_value):
                bad_index.append(index2)
                if raise_exception:
                    raise ValueError

    # remove rows from scores_df that aren't in trials_df
    for index, row in scores_df.iterrows():
        if trials_df[trials_df["number"] == row["trial"]].shape[0] == 0:
            print(f"orphan scores_df trial {row['trial']}")
            bad_index.append(index)
            if raise_exception:
                raise ValueError
    scores_df = scores_df.drop(bad_index)
    return scores_df


def assert_trials_and_scores_csv_match(exp_path, trials_df, scores_df):
    best_value = trials_df.iloc[0]["value"].item()
    best_exp_num = trials_df.iloc[0]["number"].item()
    scores_df = drop_bad_rows_from_scores_csv(trials_df, scores_df, False)
    scores_df = drop_bad_rows_from_scores_csv(trials_df, scores_df, True)
    best_exp_data = filter_by_trial(scores_df, best_exp_num)
    assert best_exp_data.shape[0] == 1
    return best_value, best_exp_data, best_exp_num, scores_df


def assert_score_from_row_is_correct(exp_path, exp_data, exp_num, key):
    metric = exp_data[key].item()
    logs_path = os.path.join(exp_path, str(exp_num), "stats")

    correct_info = {}
    for info in ["epochs", "score_history", "raw_score_history"]:
        log_path = os.path.join(
            logs_path,
            f"stat_getter_MultipleValidators_{key}_AccuracyValidator_{info}.npy",
        )
        correct_info[info] = np.load(log_path)

    assert np.array_equal(
        correct_info["score_history"], correct_info["raw_score_history"]
    )

    config = get_commandline_config(exp_path, exp_num)
    validator_filename = get_validator_filename(config["validator"])
    validator_best = os.path.join(
        logs_path,
        f"{validator_filename}_best.json",
    )
    with open(validator_best, "r") as f:
        validator_best = json.load(f)

    best_idx = np.where(correct_info["epochs"] == validator_best["best_epoch"])[0]
    logged_metric = correct_info["score_history"][best_idx][0]

    if not np.isclose(metric, logged_metric):
        raise ValueError
    return metric


def get_trials_csv(e):
    filename = os.path.join(e, "trials.csv")
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df.sort_values(by=["value", "number"], ascending=[False, True]).dropna()
    return None


def get_reproductions_csv(e):
    filename = os.path.join(e, "reproduction_score_vs_test_accuracy.csv")
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        return df.sort_values(
            by=["validation_score", "trial"], ascending=[False, True]
        ).dropna()
    return None


def get_scores_csv(e):
    filename = os.path.join(e, "score_vs_test_accuracy.csv")
    df = pd.read_csv(filename)
    return df.sort_values(
        by=["validation_score", "trial"], ascending=[False, True]
    ).dropna()


def get_best_trial_json(e):
    filename = os.path.join(e, "best_trial.json")
    with open(filename, "r") as f:
        all_data = json.load(f)
        best_val_score = json.loads(all_data["_values"])[0]
        exp_num = json.loads(all_data["number"])
    return best_val_score, exp_num


def assert_best_matches_json(e, best_val_score, exp_num=None):
    best_val_score_json, best_exp_num_json = get_best_trial_json(e)
    assert np.isclose(best_val_score, best_val_score_json)
    assert exp_num == best_exp_num_json


def get_trials_and_scores_csv(e, assert_best_trial_json=False):
    trials_df = get_trials_csv(e)
    if trials_df_is_not_empty(trials_df):
        scores_df = get_scores_csv(e)
        (
            best_val_score,
            exp_data,
            exp_num,
            scores_df,
        ) = assert_trials_and_scores_csv_match(e, trials_df, scores_df)
        if assert_best_trial_json:
            assert_best_matches_json(e, best_val_score, exp_num)
    else:
        scores_df, best_val_score, exp_data, exp_num = [None] * 4
    return trials_df, scores_df, best_val_score, exp_data, exp_num


def best_trial(e):
    trials_df, scores_df, best_val_score, exp_data, exp_num = get_trials_and_scores_csv(
        e
    )
    if trials_df_is_not_empty(trials_df):
        target_train_acc = assert_score_from_row_is_correct(
            e, exp_data, exp_num, "target_train_acc_class_avg"
        )
        return best_val_score, target_train_acc, scores_df.shape[0]
    return None, None, None

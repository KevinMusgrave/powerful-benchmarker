import unittest

import numpy as np
import pandas as pd

from validator_tests.utils.threshold_utils import (
    TARGET_ACCURACY,
    get_avg_top_n_acc_by_group,
    group_by_task,
    group_by_task_validator,
)


class TestThresholdUtils(unittest.TestCase):
    def get_avg_top_n_acc_by_group_helper(self, sort_by, new_col_name):
        validators = ["A", "B", "C"]
        validator_args_list = ["x", "y", "z"]
        adapters = ["DANN", "MCD", "MMD"]
        nlargest_list = [1, 10, 50]
        size = 100

        df = {
            "validator": np.random.choice(validators, size=(size,)),
            "validator_args": np.random.choice(validator_args_list, size=(size,)),
            "dataset": ["mnist"] * size,
            "src_domains": [("mnist",)] * size,
            "target_domains": [("mnistm",)] * size,
        }
        df = pd.DataFrame.from_dict(df)
        dfs = []
        for adapter in adapters:
            dfs.append(df.copy().assign(adapter=adapter))
        df = pd.concat(dfs)

        df = df.assign(
            **{
                "score": np.random.randn(len(df)),
                TARGET_ACCURACY: np.random.randn(len(df)),
            }
        )

        # group_by_task(False)
        for nlargest in nlargest_list:
            x = get_avg_top_n_acc_by_group(
                df, group_by_task(False), nlargest, sort_by, new_col_name
            )
            direct_compute = df.nlargest(nlargest, sort_by)[TARGET_ACCURACY].mean()
            self.assertTrue(np.isclose(x[new_col_name].item(), direct_compute))

        # group_by_task(True)
        for nlargest in nlargest_list:
            x = get_avg_top_n_acc_by_group(
                df, group_by_task(True), nlargest, sort_by, new_col_name
            )

            correct_means = {}
            for adapter in adapters:
                curr_df = df[df["adapter"] == adapter]
                curr_df = curr_df.sort_values([sort_by], ascending=False)
                correct_means[adapter] = curr_df.head(nlargest)[TARGET_ACCURACY].mean()

            for k, v in correct_means.items():
                self.assertTrue(
                    np.isclose(x[x["adapter"] == k][new_col_name].item(), v)
                )

        # group_by_task_validator(False)
        for nlargest in nlargest_list:
            x = get_avg_top_n_acc_by_group(
                df,
                group_by_task_validator(False),
                nlargest,
                sort_by,
                new_col_name,
            )

            correct_means = {}
            for validator in validators:
                for validator_args in validator_args_list:
                    curr_df = df[
                        (df["validator"] == validator)
                        & (df["validator_args"] == validator_args)
                    ]
                    curr_df = curr_df.sort_values([sort_by], ascending=False)
                    correct_means[f"{validator}.{validator_args}"] = curr_df.head(
                        nlargest
                    )[TARGET_ACCURACY].mean()

            for k, v in correct_means.items():
                validator, validator_args = k.split(".")
                self.assertTrue(
                    np.isclose(
                        x[
                            (x["validator"] == validator)
                            & (x["validator_args"] == validator_args)
                        ][new_col_name].item(),
                        v,
                    )
                )

        # group_by_task_validator(True)
        for nlargest in nlargest_list:
            x = get_avg_top_n_acc_by_group(
                df,
                group_by_task_validator(True),
                nlargest,
                sort_by,
                new_col_name,
            )

            correct_means = {}
            for validator in validators:
                for validator_args in validator_args_list:
                    for adapter in adapters:
                        curr_df = df[
                            (df["validator"] == validator)
                            & (df["validator_args"] == validator_args)
                            & (df["adapter"] == adapter)
                        ]
                        curr_df = curr_df.sort_values([sort_by], ascending=False)
                        correct_means[
                            f"{validator}.{validator_args}.{adapter}"
                        ] = curr_df.head(nlargest)[TARGET_ACCURACY].mean()

            for k, v in correct_means.items():
                validator, validator_args, adapter = k.split(".")
                self.assertTrue(
                    np.isclose(
                        x[
                            (x["validator"] == validator)
                            & (x["validator_args"] == validator_args)
                            & (x["adapter"] == adapter)
                        ][new_col_name].item(),
                        v,
                    )
                )

    def test_get_avg_top_n_acc_by_group(self):
        self.get_avg_top_n_acc_by_group_helper(TARGET_ACCURACY, "best_acc")
        self.get_avg_top_n_acc_by_group_helper("score", "predicted_best_acc")

from ..utils import constants as const, common_functions as c_f
from collections import defaultdict
from scipy import stats as scipy_stats

class BaseAggregator:
    def __init__(self, split_to_aggregate):
        self.meta_accuracies = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.split_to_aggregate = split_to_aggregate

    def update_accuracies(self, split_scheme_name, splits_to_eval, hooks, tester):
        for split in splits_to_eval:
            untrained_trunk_accuracies = hooks.get_accuracies_of_epoch(tester, split, const.UNTRAINED_TRUNK_INT)
            untrained_trunk_embedder_accuracies = hooks.get_accuracies_of_epoch(tester, split, const.UNTRAINED_TRUNK_AND_EMBEDDER_INT)
            best_split_accuracies, _ = hooks.get_accuracies_of_best_epoch(tester, split, ignore_epoch=const.IGNORE_ALL_UNTRAINED)
            accuracies_dict = {const.UNTRAINED_TRUNK: untrained_trunk_accuracies, const.UNTRAINED_TRUNK_AND_EMBEDDER: untrained_trunk_embedder_accuracies, const.TRAINED: best_split_accuracies}
            for trained_status, accuracies in accuracies_dict.items():
                if len(accuracies) > 0:
                    accuracy_keys = [k for k in accuracies[0].keys() if any(acc in k for acc in tester.accuracy_calculator.get_curr_metrics())]
                    for k in accuracy_keys:
                        self.meta_accuracies[split][trained_status][k][split_scheme_name] = accuracies[0][k]


    def record_accuracies(self, splits_to_eval, meta_record_keeper, hooks, tester):
        record_keeper_group_names = self.get_eval_record_name_dict(hooks, tester, splits_to_eval)
        if len(self.meta_accuracies) > 0:
            for split in splits_to_eval:
                group_name = record_keeper_group_names[split]
                len_of_existing_records = c_f.try_getting_db_count(meta_record_keeper, group_name)
                for trained_status, accuracies in self.meta_accuracies[split].items():
                    if len(accuracies) > 0:
                        len_of_existing_records += 1
                        averages = {k: self.get_aggregate_performance(v)  for k, v in accuracies.items()}
                        meta_record_keeper.update_records(averages, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        if len(c_f.first_val_of_dict(accuracies)) > 1:
                            standard_errors = {"SEM_%s"%k: scipy_stats.sem(list(v.values())) for k, v in accuracies.items()}
                            meta_record_keeper.update_records(standard_errors, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        meta_record_keeper.update_records({const.TRAINED_STATUS_COL_NAME: trained_status}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
                        meta_record_keeper.update_records({"timestamp": c_f.get_datetime()}, global_iteration=len_of_existing_records, input_group_name_for_non_objects=group_name)
            meta_record_keeper.save_records()


    def get_accuracy_and_standard_error(self, hooks, tester, meta_record_keeper, num_split_schemes, split_name=None):
        if split_name is None:
            split_name = self.split_to_aggregate
        group_name = self.get_eval_record_name_dict(hooks, tester, [split_name])[split_name]
        def get_average_best_and_sem(key):
            if num_split_schemes > 1:
                sem_key = "SEM_%s"%key
                columns = "%s, %s"%(key, sem_key)
                return_keys = (key, sem_key)
            else:
                columns = key
                return_keys = (key, )
            query = "SELECT {0} FROM {1} WHERE {2}=? AND id=(SELECT MAX(id) FROM {1})".format(columns, group_name, const.TRAINED_STATUS_COL_NAME)
            return meta_record_keeper.query(query, values=(const.TRAINED,), use_global_db=False), return_keys
        q, keys = hooks.try_primary_metric(tester, get_average_best_and_sem)
        if len(keys) > 1:
            return tuple(q[0][k] for k in keys)
        return q[0][keys[0]]


    def get_eval_record_name_dict(self, hooks, tester, split_names):
        base_output = c_f.get_eval_record_name_dict(hooks, tester, split_names=split_names)
        return {k:"{}_{}".format(self.__class__.__name__, v) for k,v in base_output.items()}

    def get_aggregate_performance(self, accuracy_per_split):
        raise NotImplementedError
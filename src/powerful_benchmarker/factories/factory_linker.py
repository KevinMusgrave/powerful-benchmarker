

# class FactoryLink:
#     def __init__(self, api_parser, factories):
#         self.api_parser = api_parser
#         self.args = self.api_parser.args
#         self.factories = factories

#     def get_additional_kwargs(self, name):


#     def set_sampler(self):
#         ak = {"labels": self.api_parser.split_manager.get_labels("train", "train"),
#             "dataset_length": len(self.api_parser.split_manager.get_dataset("train", "train"))}
#         self.api_parser.sampler = self.factories["sampler"].create(self.args.sampler, additional_kwargs=ak)
               
#     def set_loss_function(self):
#         num_classes = self.api_parser.split_manager.get_num_labels("train", "train")
#         ak = {k: {"num_classes": num_classes} for k in self.args.loss_funcs.keys()}
#         self.api_parser.loss_funcs = self.factories["loss"].create(named_specs=self.args.loss_funcs, num_classes, additional_kwargs=ak)

#     def set_mining_function(self):
#         self.api_parser.mining_funcs = self.factories["miner"].create(named_specs=self.args.mining_funcs)

#     def set_model(self):
#         self.api_parser.models = self.factories["model"].create(named_specs=self.args.models)


#     def set_tester(self):
#         ak = {"data_device": self.api_parser.device,
#                             "data_and_label_getter": self.api_parser.split_manager.data_and_label_getter,
#                             "end_of_testing_hook": self.api_parser.hooks.end_of_testing_hook}
#         self.api_parser.tester = self.factories["tester"].create(self.args.tester, additional_kwargs=ak)

#     def set_trainer(self):
#         x = self.api_parser
#         ak = {
#             "models": x.models,
#             "optimizers": x.optimizers,
#             "sampler": x.sampler,
#             "collate_fn": x.get_collate_fn(),
#             "loss_funcs": x.loss_funcs,
#             "mining_funcs": x.mining_funcs,
#             "dataset": x.split_manager.get_dataset("train", "train", log_split_details=True),
#             "data_device": x.device,
#             "lr_schedulers": x.lr_schedulers,
#             "gradient_clippers": x.gradient_clippers,
#             "data_and_label_getter": x.split_manager.data_and_label_getter,
#             "dataset_labels": list(x.split_manager.get_label_set("train", "train")),
#             "end_of_iteration_hook": x.hooks.end_of_iteration_hook,
#             "end_of_epoch_hook": x.get_end_of_epoch_hook()
#         }
#         self.api_parser.trainer = self.factories["trainer"].create(self.args.trainer, additional_kwargs=ak)
    
#     def set_optimizers(self):
#         self.api_parser.optimizers, self.api_parser.lr_schedulers, self.api_parser.gradient_clippers = \ 
#             self.factories["optimizer"].create(named_specs=self.args.optimizers, additional_kwargs=ak)
        
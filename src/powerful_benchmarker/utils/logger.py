from pytorch_adapt.frameworks.ignite.loggers import (
    BasicLossLogger,
    IgniteRecordKeeperLogger,
)


class Logger:
    def __init__(self, folder):
        self.logger1 = BasicLossLogger()
        self.logger2 = IgniteRecordKeeperLogger(folder=folder)

    def add_training(self, *args, **kwargs):
        return self.logger1.add_training(*args, **kwargs)

    def add_validation(self, *args, **kwargs):
        self.logger2.add_validation(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.logger2.write(*args, **kwargs)

    def get_losses(self):
        return self.logger1.get_losses()

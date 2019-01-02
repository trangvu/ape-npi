import numpy as np
from translate import utils
from translate.translation_model import TranslationModel


class MultiTaskModel:
    def __init__(self, tasks, **kwargs):
        self.models = []
        self.ratios = []

        for i, task in enumerate(tasks, 1):
            if task.name is None:
                task.name = 'task_{}'.format(i)

            # merging both dictionaries (task parameters have a higher precedence)
            kwargs_ = dict(**kwargs)
            kwargs_.update(task)
            model = TranslationModel(**kwargs_)

            self.models.append(model)
            self.ratios.append(task.ratio if task.ratio is not None else 1)

        self.main_model = self.models[0]
        self.ratios = [ratio / sum(self.ratios) for ratio in self.ratios]  # unit normalization

    def train(self, **kwargs):
        for model in self.models:
            utils.log('initializing {}'.format(model.name))
            model.init_training(**kwargs)

        utils.log('starting training')
        while True:
            i = np.random.choice(len(self.models), 1, p=self.ratios)[0]
            model = self.models[i]
            try:
                model.train_step(**kwargs)
            except (utils.FinishedTrainingException, KeyboardInterrupt):
                utils.log('exiting...')
                self.main_model.save()
                return
            except utils.EvalException:
                if i == 0:
                    model.save()
                    step, score = model.training.scores[-1]
                    model.manage_best_checkpoints(step, score)
            except utils.CheckpointException:
                if i == 0:   # only save main model (includes all variables)
                    model.save()
                    step, score = model.training.scores[-1]
                    model.manage_best_checkpoints(step, score)

    def decode(self, *args, **kwargs):
        self.main_model.decode(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.main_model.evaluate(*args, **kwargs)

    def align(self, *args, **kwargs):
        self.main_model.align(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        self.main_model.initialize(*args, **kwargs)

    def save(self, *args, **kwargs):
        self.main_model.save(*args, **kwargs)

    def save_embedding(self, *args, **kwargs):
        self.main_model.save_embedding(*args, **kwargs)

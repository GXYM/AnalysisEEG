import torch
import torch.nn as nn
import torch.nn.functional as F


class SleepModel(nn.Module):
    def __init__(self, num_classes, is_training=True):
        super().__init__()
        self.is_training = is_training
        self.predoctor = torch.nn.Sequential(
                                    nn.Linear(4, 512),
                                    nn.BatchNorm1d(512),
                                    nn.Softsign(),
                                    nn.Linear(512, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.Softsign(),
                                    nn.Linear(1024, 512),
                                    nn.BatchNorm1d(512),
                                    nn.Dropout(0.5),
                                    nn.PReLU(),
                                    nn.Linear(512, num_classes))

    def load_model(self, model_path):
        print('Loading from {}'.format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict['model'])

    def forward(self, x):
        x = self.predoctor(x)

        return F.log_softmax(x, dim=1)





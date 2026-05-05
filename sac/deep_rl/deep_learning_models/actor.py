import numpy as np
import torch as th

from deep_learning_models.feature_extractor import DLinearFeatureExtractor, GRUFeatureExtractor, LinearFeatureExtractor, TransformerFeatureExtractor

class ContinuousActor(th.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(ContinuousActor, self).__init__()
        self.actions_base = th.nn.Sequential(LinearFeatureExtractor(input_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                            th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())
        self.actions_mean = th.nn.Linear(hidden_dim, output_dim)
        self.actions_log_std = th.nn.Linear(hidden_dim, output_dim)

        action_lo = np.asarray(action_range[0], dtype=np.float32)
        action_hi = np.asarray(action_range[1], dtype=np.float32)
        self.register_buffer('ACTION_SCALE', th.from_numpy(0.5 * (action_hi - action_lo)))
        self.register_buffer('ACTION_BIAS', th.from_numpy(0.5 * (action_hi + action_lo)))

        self.LOG_STD_MIN = log_std_range[0]
        self.LOG_STD_MAX = log_std_range[1]

    def forward(self, x):
        x = self.actions_base(x)
        actions_std = th.clamp(self.actions_log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return th.distributions.Normal(self.actions_mean(x), actions_std)

    def sample_log_prob(self, x):
        dist = self.forward(x)
        x = dist.rsample()
        y = th.tanh(x)
        action = self.ACTION_SCALE * y + self.ACTION_BIAS
        log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True) - th.log(self.ACTION_SCALE * (1 - y.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    @th.no_grad()
    def act(self, x):
        return self.sample_log_prob(x)[0]

    @th.no_grad()
    def get_deterministic_action(self, x):
        x = self.actions_base(x)
        x = th.tanh(self.actions_mean(x))
        return x * self.ACTION_SCALE + self.ACTION_BIAS


class GRUActor(ContinuousActor):

    def __init__(self, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(GRUActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(GRUFeatureExtractor(input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())


class TransformerActor(ContinuousActor):
    def __init__(self, seq_len, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(TransformerActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(TransformerFeatureExtractor(seq_len, input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())


class DLinearActor(ContinuousActor):
    def __init__(self, seq_len, input_dim, output_dim, hidden_dim, action_range, log_std_range=(-20, 2)):
        super(DLinearActor, self).__init__(input_dim, output_dim, hidden_dim, action_range, log_std_range)
        self.actions_base = th.nn.Sequential(DLinearFeatureExtractor(seq_len, input_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU(),
                                             th.nn.Linear(hidden_dim, hidden_dim), th.nn.LeakyReLU())

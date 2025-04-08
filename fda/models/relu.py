from torch import nn


class ReluLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        sigma0=10.0,
        trainable=True,
        batchnorm=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))
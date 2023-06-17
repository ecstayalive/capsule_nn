import torch
from torch import Tensor
from torch.nn import Parameter, Module, init
import torch.nn.functional as F
from torch.autograd import Variable
from ..functional import squash

__all__ = ["CapsLinear"]


class CapsLinear(Module):
    r"""The linear capsule layer. It is similar to Linear layer in pytorch.

    Args:
        in_features: input feature tensor
        out_features: output feature tensor
        routings: the times of doing routings
            Default: ``3``

    Shape:
        - Input: :math:`(*, H_{in\_feature\_numbers}, H_{in\_dimension})` where :math:`*` means any number of
          dimensions including none, at the same time :math:`H_{in\_feature\_numbers} = \text{number of input features}`
          and :math:`H_{in\_dimension} = \text{input dimension}`.
        - Output: :math:`(*, H_{out\_feature\_numbers}, H_{out\_dimension})` where :math:`*` means any number of
          dimensions including none, at the same time :math:`H_{out\_feature\_numbers} = \text{number of output features}`
          and :math:`H_{out\_dimension} = \text{output dimension}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(*, out\_feature\_numbers, in\_feature\_numbers, out\_dimension, in\_dimension)`
            where :math:`*` means any number of dimensions.
            The values are initialized from :math:`\alpha * \mathcal{N}(0, 1)`, where
            :math:`\alpha` mean a scaling factor.

    Examples::

        >>> m = CapsLinear((10, 8), (5, 4))
        >>> input = torch.randn(1, 10, 8)
        >>> output = m(input)
        >>> print(output.shape)
        torch.Size([1, 5, 4])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: Tensor
    out_features: Tensor
    routings: int
    weight: Tensor

    def __init__(self, in_features: tuple, out_features: tuple, routings: int = 3,
                 device=None, dtype=None) -> None:
        super(CapsLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self._weight_shape = (out_features[0], in_features[0],
                              out_features[1], in_features[1])
        self.weight = Parameter(
            torch.empty(self._weight_shape, **factory_kwargs)
        )
        self.routings = routings
        self.reset_parameters()

    def reset_parameters(self, scales: float = 1.) -> None:
        init.xavier_normal_(self.weight) * scales

    def forward(self, input: Tensor) -> Tensor:
        x_hat = torch.matmul(self.weight, input[:, None, :, :, None])
        x_hat = torch.squeeze(x_hat, dim=-1)
        x_hat_detached = x_hat.detach()

        b = Variable(
            torch.zeros(input.shape[0], self.out_features[0],
                        self.in_features[0], device=self.weight.device)
        )
        assert self.routings > 0, "The parameter 'routings' should be > 0."
        for idx in range(self.routings):
            c = F.softmax(b, dim=1)
            if idx == self.routings - 1:
                output = squash(
                    torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True)
                )
            else:
                output = squash(
                    torch.sum(c[:, :, :, None] * x_hat_detached,
                              dim=-2, keepdim=True)
                )
                b = b + torch.sum(output * x_hat_detached, dim=-1)

        return torch.squeeze(output, dim=-2)

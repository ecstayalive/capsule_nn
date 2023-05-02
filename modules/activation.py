from torch import Tensor
from torch.nn import Module

from .. import functional as F

__all__ = ["Squash"]


class Squash(Module):
    r"""Non-Linear activation function used in Capsule

    Args:
        None

    Shape:
        - Input: :math:`(*)`, where :math:`*` means that any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples:

    >>> m = Squash()
    >>> input = torch.randn(1, 10, 1)
    >>> output = m(input)
    >>> print(output.shape)
    torch.Size([1, 10, 1])

    """

    def __init__(self):
        super(Squash, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.squash(input)

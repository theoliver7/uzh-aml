import torch
import torch.nn as nn
from torch.autograd import Function

def scatter_sort(x, batch, fill_value=-1e16):
    num_nodes = torch.bincount(batch)
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch]) + (batch * num_nodes.max().item())

    dense_x = torch.full((batch.size(0) * num_nodes.max().item(),), fill_value, device=x.device)
    dense_x[index] = x
    dense_x = dense_x.view(-1, num_nodes.max().item())

    sorted_x, _ = dense_x.sort(dim=-1, descending=True)
    cumsum_sorted_x = sorted_x.cumsum(dim=-1)
    cumsum_sorted_x = cumsum_sorted_x.view(-1)

    sorted_x = sorted_x.view(-1)
    filled_index = sorted_x != fill_value

    sorted_x = sorted_x[filled_index]
    cumsum_sorted_x = cumsum_sorted_x[filled_index]

    return sorted_x, cumsum_sorted_x


def _make_ix_like(batch):
    num_nodes = torch.bincount(batch)
    idx = [torch.arange(1, i + 1, dtype=torch.long, device=batch.device) for i in num_nodes]
    idx = torch.cat(idx, dim=0)

    return idx


def _threshold_and_support(x, batch):
    num_nodes = torch.bincount(batch)
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    sorted_input, input_cumsum = scatter_sort(x, batch)
    input_cumsum = input_cumsum - 1.0
    rhos = _make_ix_like(batch).to(x.dtype)
    support = rhos * sorted_input > input_cumsum

    support_size = torch.bincount(batch, support.to(batch.dtype))
    idx = support_size + cum_num_nodes - 1
    mask = idx < 0
    idx[mask] = 0
    tau = input_cumsum.gather(0, idx)
    tau /= support_size.to(x.dtype)

    return tau, support_size


class SparsemaxFunction(Function):

    @staticmethod
    def forward(ctx, x, batch):
        max_val, _ = torch.max(x.new_zeros(x.size(0)), dim=0)
        for i in range(x.size(0)):
            max_val[batch == batch[i]], _ = torch.max(x[batch == batch[i]], dim=0)
        x -= max_val[batch]
        tau, supp_size = _threshold_and_support(x, batch)
        output = torch.clamp(x - tau[batch], min=0)
        ctx.save_for_backward(supp_size, output, batch)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0

        v_hat = torch.zeros_like(grad_input)
        for i in range(grad_input.size(0)):
            v_hat[batch == batch[i]] += torch.scatter_add(torch.zeros_like(grad_input), dim=0,
                                                           index=batch[batch == batch[i]],
                                                           src=grad_input[batch == batch[i]]) / supp_size[
                                                  batch == batch[i]]

        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)

        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):

    def __init__(self):
        super(Sparsemax, self).__init__()

    def forward(self, x, batch):
        return sparsemax(x, batch)


if __name__ == '__main__':
    sparse_attention = Sparsemax()
    input_x = torch.tensor([1.7301, 0.6792, -1.0565, 1.6614, -0.3196, -0.7790, -0.3877, -0.4943, 0.1831, -0.0061])
    input_batch = torch.cat([torch.zeros(4, dtype=torch.long), torch.ones(6, dtype=torch.long)], dim=0)
    res = sparse_attention(input_x, input_batch)
    print(res)

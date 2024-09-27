from torch.nn.functional import softmin

dims = q.size(0)
softmin((q[:, None].expand((-1, dims)) - k[None, :].expand((dims, -1))).abs(),dim=-1) @ v

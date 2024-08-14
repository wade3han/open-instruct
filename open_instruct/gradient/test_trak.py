import torch
from test_trak.projectors import CudaProjector, ProjectionType

block_size = 128

number_of_params = 32
seed = 0

device = torch.device('cuda')
dtype = torch.float32
projector_batch_size = 128

projector = CudaProjector(grad_dim=number_of_params,
                          proj_dim=8192,
                          seed=seed,
                          device=device,
                          dtype=dtype,
                          block_size=block_size,
                          max_batch_size=projector_batch_size,
                          proj_type=ProjectionType.rademacher)

input_tensor = torch.randn((1, number_of_params)).to(device=device, dtype=dtype)
projected_tensor = projector.project(input_tensor, model_id=0)
print(projected_tensor)

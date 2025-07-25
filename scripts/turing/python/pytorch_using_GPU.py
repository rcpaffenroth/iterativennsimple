import torch

if torch.cuda.is_available():
    print('CUDA available')
    print('current device', torch.cuda.current_device())
    print('device count', torch.cuda.device_count())
    print('device name', torch.cuda.get_device_name(0))
    print('arch list', torch.cuda.get_arch_list())
else:
    print('CUDA *not* available')

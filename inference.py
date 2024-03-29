import os
import sys

import torch
import torch.nn as nn
from thop import profile


def count_params_and_flops(model, device):
    img_size = torch.randn(1, 3, 224, 224).to(device)
    audio_size = torch.randn(1, 40, 41).to(device)

    model.load_state_dict(torch.load('runs/test/best_model.pt', map_location=device))
    model.eval()

    flops, params = profile(model, inputs=(audio_size, img_size, None, ))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("Params=", str(params / 1e6) + '{}'.format("M"))

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.4fM" % (total / 1e6))
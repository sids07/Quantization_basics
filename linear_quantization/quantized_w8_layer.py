import torch
import torch.nn as nn
from linear_quantization.helper import w8_a16_forward

class W8A16LinearLayer(nn.Module):
    
    def __init__(self, in_features, out_features,
                    bias=True, dtype=torch.float32):
        """Here, by default torch doesn't allow to initialize Parameter with tensor of integer datatype.
        So, we register them as a buffer.

        Args:
            in_features (int): represents number of input feature being passed to layer
            out_features (int): represents number of output features from specific layer
            bias (bool, optional): Whether to provide bias or not. Defaults to True.
            dtype (_type_, optional): Precision of weights. Defaults to torch.float32.
        """
        super().__init__()

        self.register_buffer(
            "int8_weights",
            torch.randint(
                -128, 127, (out_features, in_features), dtype=torch.int8
            )
        )

        self.register_buffer("scales",
                            torch.randn((out_features), dtype=dtype))

        if bias:
            self.register_buffer("bias",
                                    torch.randn((1, out_features),
                                                dtype=dtype))

        else:
            self.bias = None

    def quantize(self, weights):
        """This method is used to quantize model's weight in 8-bit Integer.

        Args:
            weights (numpy.array): Orginal weights of the models
        """
        w_fp32 = weights.clone().to(torch.float32)

        scales = w_fp32.abs().max(dim=-1).values / 127
        scales = scales.to(weights.dtype)

        int8_weights = torch.round(weights
                        /scales.unsqueeze(1)).to(torch.int8)

        self.int8_weights = int8_weights
        self.scales = scales

    def forward(self, input):
        return w8_a16_forward(self.int8_weights,
                            input, self.scales, self.bias)

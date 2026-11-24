import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def unpack_int4_rows(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack [in_features // 8, out_features] int32 -> [in_features, out_features] uint8."""
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    unpacked = (qweight.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF
    return unpacked.reshape(-1, qweight.shape[1]).to(torch.uint8)


def unpack_int4_cols(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack [num_groups, out_features // 8] int32 -> [num_groups, out_features] int32."""
    shifts = torch.arange(0, 32, 4, device=qzeros.device, dtype=torch.int32)
    unpacked = (qzeros.unsqueeze(2) >> shifts.view(1, 1, 8)) & 0xF
    return unpacked.reshape(qzeros.shape[0], -1)


class QALoRALinear(nn.Module):
    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int = 32,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features = qweight.shape[0] * 8
        self.out_features = qweight.shape[1]
        self.group_size = group_size
        self.num_groups = self.in_features // group_size
        self.r = r
        self.scaling = lora_alpha / r
        self.merged = False

        # Frozen 4-bit packed base weights & group quantization parameters
        self.register_buffer("qweight", qweight.detach().clone())
        self.register_buffer("scales", scales.detach().clone())
        self.register_buffer("zeros", zeros.detach().clone())
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

        # Trainable group-pooled LoRA adapters: A maps num_groups -> r, B maps r -> out_features
        dtype = scales.dtype
        device = scales.device
        self.lora_A = nn.Parameter(torch.empty((r, self.num_groups), dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), dtype=dtype, device=device))
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_gptq_layer(cls, gptq_layer: nn.Module, r: int = 16, lora_alpha: int = 32):
        qweight = gptq_layer.qweight.data
        scales = gptq_layer.scales.data.to(torch.bfloat16)
        group_size = getattr(gptq_layer, "group_size", 32)
        bias = gptq_layer.bias.data.to(torch.bfloat16) if getattr(gptq_layer, "bias", None) is not None else None

        # Unpack qzeros and auto-detect GPTQ v1 (zero - 1) vs v2 (exact zero)
        raw_zeros = unpack_int4_cols(gptq_layer.qzeros.data).to(torch.bfloat16)
        zeros_v1 = raw_zeros + 1.0
        zeros_v2 = raw_zeros

        with torch.no_grad():
            test_x = torch.randn(1, qweight.shape[0] * 8, device=qweight.device, dtype=gptq_layer.scales.dtype)
            ref_y = gptq_layer(test_x).to(torch.bfloat16)

            w_int = unpack_int4_rows(qweight).to(torch.bfloat16)
            w_v1 = (w_int.view(-1, group_size, scales.shape[1]) - zeros_v1.unsqueeze(1)) * scales.unsqueeze(1)
            w_v2 = (w_int.view(-1, group_size, scales.shape[1]) - zeros_v2.unsqueeze(1)) * scales.unsqueeze(1)

            err_v1 = (F.linear(test_x.to(torch.bfloat16), w_v1.reshape(-1, scales.shape[1]).T, bias) - ref_y).abs().mean()
            err_v2 = (F.linear(test_x.to(torch.bfloat16), w_v2.reshape(-1, scales.shape[1]).T, bias) - ref_y).abs().mean()
            zeros = zeros_v1 if err_v1 < err_v2 else zeros_v2

        return cls(
            qweight=qweight,
            scales=scales,
            zeros=zeros,
            bias=bias,
            group_size=group_size,
            r=r,
            lora_alpha=lora_alpha,
        )

    def _dequantize_weight(self, target_dtype: torch.dtype) -> torch.Tensor:
        w_int = unpack_int4_rows(self.qweight).to(target_dtype)
        w_grouped = w_int.view(self.num_groups, self.group_size, self.out_features)
        s = self.scales.to(target_dtype).unsqueeze(1)
        z = self.zeros.to(target_dtype).unsqueeze(1)
        return ((w_grouped - z) * s).reshape(self.in_features, self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_cast = x.to(self.scales.dtype)

        # Base INT4 dequantized projection
        w = self._dequantize_weight(x_cast.dtype)
        bias = self.bias.to(x_cast.dtype) if self.bias is not None else None
        out = F.linear(x_cast, w.T, bias)

        if not self.merged:
            # Group-wise average pooling across each 32-channel block
            x_pool = x_cast.reshape(*x_cast.shape[:-1], self.num_groups, self.group_size).mean(dim=-1)
            lora_out = F.linear(F.linear(self.dropout(x_pool), self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out.to(orig_dtype)

    @torch.no_grad()
    def merge(self):
        """Losslessly fold group-pooled LoRA weights into the group zero-points."""
        if self.merged:
            return
        # Effective per-weight update for each group: (B @ A)^T * (scaling / group_size)
        delta_per_weight = (self.lora_B @ self.lora_A).T * (self.scaling / self.group_size)
        self.zeros.copy_(self.zeros - (delta_per_weight / self.scales))
        self.merged = True


def inject_qalora(model: nn.Module, r: int = 16, lora_alpha: int = 32) -> int:
    """Freeze base model and replace target GPTQ linear layers with QALoRALinear."""
    for param in model.parameters():
        param.requires_grad = False

    replaced = 0
    for name, module in list(model.named_modules()):
        if any(name.endswith(f".{t}") for t in TARGET_MODULES) and hasattr(module, "qweight"):
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, QALoRALinear.from_gptq_layer(module, r=r, lora_alpha=lora_alpha))
            replaced += 1

    return replaced
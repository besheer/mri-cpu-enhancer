import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        dw_channel = c * 2
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = c * 2
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = nn.GroupNorm(1, c)
        self.norm2 = nn.GroupNorm(1, c)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * 0.1

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * 0.1

class NAFNetLite(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, width=16, num_blocks=3):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.ending = nn.Conv2d(width, out_channels, 3, padding=1, bias=True)

    def forward(self, x):
        feat = self.intro(x)
        feat = self.body(feat)
        noise = self.ending(feat)
        # Residual learning: الصورة الأصلية مطروحاً منها الضوضاء المستخرجة
        return torch.clamp(x - noise * 0.5, 0.0, 1.0)

model = NAFNetLite()
model.eval()
dummy = torch.randn(1, 1, 256, 256)
torch.onnx.export(
    model, dummy, "models/swift_nafnet_03t.onnx",
    opset_version=14,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}}
)
print("✓ تم تصدير النموذج swift_nafnet_03t.onnx بنجاح!")

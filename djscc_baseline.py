"""DeepJSCC baseline for STAR, on the same 5G NR OFDM link as STAR.

A standard deep joint source-channel coding (DeepJSCC) autoencoder in the style of
Bourtsoulatze et al. (2019): a fully convolutional encoder maps the image to complex
channel symbols, which are placed directly on OFDM resource elements (no bits, no MCS)
and sent through the SAME Sionna link as STAR (30 kHz SCS, 14 x 128 grid, 3GPP TDL-A
100 ns, perfect CSI, LMMSE equalization, fixed Es/N0). The decoder reconstructs the
image from the equalized symbols.

Notebook integration: `transmit_djscc_x(img, esn0_db, model, link)` returns a uint8
H x W x 3 image, exactly like `transmit_baseline_x` / `transmit_sci_x`, so the speaker,
judge and S-PSNR code is unchanged.

Bandwidth ratio (complex symbols per real source value):
    rho = k / n = (c/2 * H/d * W/d) / (3 H W) = c / (6 d^2)
d = 4: c = 16 -> 1/6, c = 8 -> 1/12, c = 96 -> 1.  REs per image = k.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ─────────────────────────────── model ────────────────────────────────────────


def _conv(cin, cout, k=5, s=1):
    return nn.Conv2d(cin, cout, k, s, k // 2)


def _deconv(cin, cout, k=5, s=2):
    return nn.ConvTranspose2d(cin, cout, k, s, k // 2, output_padding=s - 1)


class DeepJSCC(nn.Module):
    """Conv/PReLU DeepJSCC.  down = 4 (two stride-2 stages) or 8 (three)."""

    def __init__(self, c=16, width=128, down=4):
        super().__init__()
        assert down in (4, 8)
        n_down = int(math.log2(down))
        enc = [_conv(3, width, 5, 2), nn.PReLU(width)]
        for _ in range(n_down - 1):
            enc += [_conv(width, width, 5, 2), nn.PReLU(width)]
        enc += [_conv(width, width), nn.PReLU(width),
                _conv(width, width), nn.PReLU(width),
                _conv(width, c)]
        dec = [_conv(c, width), nn.PReLU(width),
               _conv(width, width), nn.PReLU(width),
               _conv(width, width), nn.PReLU(width)]
        for _ in range(n_down - 1):
            dec += [_deconv(width, width), nn.PReLU(width)]
        dec += [_deconv(width, 3), nn.Sigmoid()]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        self.c, self.width, self.down = c, width, down

    # image in [0,1], (B,3,H,W) with H, W divisible by `down`
    def encode(self, x):
        z = self.enc(x)                                   # (B, c, h, w), real
        shape = z.shape
        zc = torch.view_as_complex(z.reshape(shape[0], -1, 2).contiguous())   # (B, k)
        p = zc.abs().pow(2).mean(dim=1, keepdim=True)     # unit average energy per image,
        zc = zc / torch.sqrt(p + 1e-12)                   # as for Sionna's QAM (Es = 1)
        return zc, shape

    def decode(self, zc, shape):
        z = torch.view_as_real(zc).reshape(shape)
        return self.dec(z)

    def config(self):
        return dict(c=self.c, width=self.width, down=self.down)

    def bandwidth_ratio(self):
        return self.c / (6 * self.down ** 2)


def num_res(H, W, model):
    """Complex symbols (= OFDM REs) per H x W image, after padding to a multiple of `down`."""
    d = model.down
    Hp, Wp = math.ceil(H / d) * d, math.ceil(W / d) * d
    return (Hp // d) * (Wp // d) * model.c // 2


def save_model(model, path, **extra):
    torch.save({"config": model.config(), "state_dict": model.state_dict(), **extra}, path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = DeepJSCC(**ck["config"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


# ─────────────────────────────── channel ──────────────────────────────────────


class SionnaLink:
    """The STAR OFDM link (notebook cell 'Two-layer transmission'), carrying complex symbols.

    Each 14 x 128 grid of 1792 REs gets an independent TDL-A realization, as in STAR's
    `send_bits_esn0`. Differentiable, so it can also be used during training.
    """

    def __init__(self, device="cpu", fft_size=128, num_ofdm_symbols=14, scs=30e3,
                 delay_spread=100e-9, carrier_frequency=3.5e9, grids_per_call=512):
        from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LMMSEEqualizer
        from sionna.phy.channel import GenerateOFDMChannel, ApplyOFDMChannel
        from sionna.phy.channel.tr38901 import TDL
        from sionna.phy.mimo import StreamManagement
        self.rg = ResourceGrid(num_ofdm_symbols=num_ofdm_symbols, fft_size=fft_size,
                               subcarrier_spacing=scs, num_tx=1, num_streams_per_tx=1,
                               cyclic_prefix_length=6, pilot_pattern=None,
                               num_guard_carriers=(0, 0), dc_null=False)
        sm = StreamManagement(np.array([[1]]), 1)
        tdl = TDL(model="A", delay_spread=delay_spread, carrier_frequency=carrier_frequency,
                  min_speed=0.0, max_speed=0.0)
        self.gen = GenerateOFDMChannel(tdl, self.rg, normalize_channel=True)
        self.apply = ApplyOFDMChannel()
        self.mapper = ResourceGridMapper(self.rg)
        self.eq = LMMSEEqualizer(self.rg, sm)
        self.C = self.rg.num_data_symbols
        self.device, self.grids_per_call = device, grids_per_call

    def __call__(self, zc, esn0_db):
        """zc: (B, k) complex, unit average energy. Returns equalized symbols, same shape."""
        B, k = zc.shape
        ng = math.ceil(k / self.C)
        pad = ng * self.C - k
        x = F.pad(torch.view_as_real(zc), (0, 0, 0, pad))           # pad with zero symbols
        x = torch.view_as_complex(x.contiguous()).reshape(B * ng, 1, 1, self.C)
        no = torch.tensor(10 ** (-esn0_db / 10.0), dtype=torch.float32, device=zc.device)
        out = []
        for s in range(0, x.shape[0], self.grids_per_call):
            xs = x[s:s + self.grids_per_call]
            h = self.gen(batch_size=xs.shape[0])
            y = self.apply(self.mapper(xs), h, no)
            xh, _ = self.eq(y, h, torch.zeros_like(no), no)
            out.append(xh.reshape(-1, self.C))
        return torch.cat(out).reshape(B, ng * self.C)[:, :k]


class AWGNLink:
    """Fast AWGN stand-in for training (Es = 1 per complex symbol)."""

    def __call__(self, zc, esn0_db):
        no = 10 ** (-esn0_db / 10.0)
        n = torch.randn_like(torch.view_as_real(zc)) * math.sqrt(no / 2)
        return zc + torch.view_as_complex(n.contiguous())


# ────────────────────────── notebook-facing API ───────────────────────────────


def _to_uint8(img):
    return np.array(img.convert("RGB"), np.uint8) if hasattr(img, "convert") else np.asarray(img, np.uint8)


@torch.no_grad()
def transmit_djscc_x(img, esn0_db, model, link, device=None):
    """DeepJSCC over the STAR link: PIL/uint8 image -> received uint8 H x W x 3 image."""
    device = device or next(model.parameters()).device
    npimg = _to_uint8(img)
    H, W = npimg.shape[:2]
    d = model.down
    Hp, Wp = math.ceil(H / d) * d, math.ceil(W / d) * d
    x = torch.from_numpy(npimg).permute(2, 0, 1)[None].float().div(255).to(device)
    x = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
    zc, shape = model.encode(x)
    zh = link(zc, esn0_db)
    xh = model.decode(zh, shape)[..., :H, :W]
    return (xh[0].clamp(0, 1).permute(1, 2, 0).mul(255).round().byte().cpu().numpy())


# ─────────────────────── ground-truth regions for S-PSNR ──────────────────────

# Overlay boxes of the factory generator (fractions of W, H), from generate_factory_dataset.py
FACTORY_SCREEN = (0.392, 0.214, 0.602, 0.483)
FACTORY_PLATE = (0.434, 0.536, 0.556, 0.614)


def factory_gt_mask(H, W, boxes=(FACTORY_SCREEN, FACTORY_PLATE)):
    """Ground-truth text region of a factory image (status panel + room nameplate)."""
    m = np.zeros((H, W), bool)
    for b in boxes:
        x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
        m[y1:y2, x1:x2] = True
    return m


def vehicle_gt_mask(txt_path, H, W):
    """Plate box from an OpenALPR label: 'file <TAB> x <TAB> y <TAB> w <TAB> h <TAB> PLATE'."""
    f = open(txt_path).read().strip().split("\t")
    x, y, w, h = (int(v) for v in f[1:5])
    m = np.zeros((H, W), bool)
    m[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)] = True
    return m


def psnr(a, b, mask=None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if mask is not None:
        a, b = a[mask], b[mask]
    e = np.mean((a - b) ** 2)
    return 99.0 if e == 0 else 10 * np.log10(255 ** 2 / e)

"""Train the DeepJSCC baseline.

Default: held-out factory panels 300-499 (the paper's test images come from panels 0-299),
random crops, Es/N0 drawn uniformly per step, through the STAR Sionna link.

Examples (run from the repository root):
  # factory-trained, 1/6 bandwidth ratio, over the real OFDM link
  python djscc/train_djscc.py --factory --c 16 --out djscc/djscc_c16.pt
  # generic training data (any folder of images), AWGN for speed
  python djscc/train_djscc.py --image_dir /data/DIV2K_train_HR --channel awgn --out djscc/djscc_div2k_c16.pt
"""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from djscc_baseline import DeepJSCC, SionnaLink, AWGNLink, save_model  # noqa: E402


def factory_files(root, lo, hi):
    ann = json.load(open(os.path.join(root, "factory_dataset", "annotations.json")))
    return [os.path.join(root, "factory_dataset", e["file"]) for e in ann["images"][lo:hi]]


class Crops:
    """Random crops from a list of images, kept in RAM as uint8."""

    def __init__(self, files, crop):
        self.imgs = [np.array(Image.open(f).convert("RGB"), np.uint8) for f in files]
        self.crop = crop

    def batch(self, n, device):
        out = []
        for _ in range(n):
            im = random.choice(self.imgs)
            H, W = im.shape[:2]
            y, x = random.randint(0, H - self.crop), random.randint(0, W - self.crop)
            p = im[y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                p = p[:, ::-1]
            out.append(p)
        x = torch.from_numpy(np.stack(out).copy()).permute(0, 3, 1, 2).float().div(255)
        return x.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--factory", action="store_true", help="train on factory panels [lo, hi)")
    ap.add_argument("--factory_range", type=int, nargs=2, default=[300, 500])
    ap.add_argument("--image_dir", default=None, help="folder of training images (png/jpg)")
    ap.add_argument("--c", type=int, default=16, help="latent channels (ratio = c / (6 d^2))")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--down", type=int, default=4, choices=[4, 8])
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--snr", type=float, nargs=2, default=[-8.0, 18.0], help="Es/N0 range (dB)")
    ap.add_argument("--channel", choices=["sionna", "awgn"], default="sionna")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    files = []
    if a.factory:
        files += factory_files(a.root, *a.factory_range)
    if a.image_dir:
        files += sorted(glob.glob(os.path.join(a.image_dir, "*.png")) + glob.glob(os.path.join(a.image_dir, "*.jpg")))
    assert files, "no training images (use --factory and/or --image_dir)"
    data = Crops(files, a.crop)
    print(f"{len(files)} training images | device {dev}")

    model = DeepJSCC(a.c, a.width, a.down).to(dev)
    link = SionnaLink(device=dev) if a.channel == "sionna" else AWGNLink()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7 * a.steps), int(0.9 * a.steps)], 0.3)
    print(f"DeepJSCC c={a.c} width={a.width} down={a.down} | ratio {model.bandwidth_ratio():.4f} "
          f"| {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | channel {a.channel}")

    start = 1
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        if "optim" in ck: opt.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        start = ck.get("step", 0) + 1
        print(f"resumed from {a.out} at step {start - 1}")

    t0, run = time.time(), 0.0
    for step in range(start, a.steps + 1):
        x = data.batch(a.batch, dev)
        snr = random.uniform(*a.snr)
        zc, shape = model.encode(x)
        xh = model.decode(link(zc, snr), shape)
        loss = torch.mean((xh - x) ** 2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run += loss.item()
        if step % a.log_every == 0:
            mse = run / a.log_every; run = 0.0
            print(f"step {step:6d} | mse {mse:.5f} | psnr {10*np.log10(1/max(mse,1e-12)):.2f} dB "
                  f"| {(time.time()-t0)/(step-start+1):.3f} s/step", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_model(model, a.out + ".tmp", step=step, args=vars(a),
                       optim=opt.state_dict(), sched=sched.state_dict())
            os.replace(a.out + ".tmp", a.out)          # never leave a half-written checkpoint
    print("saved", a.out)


if __name__ == "__main__":
    main()

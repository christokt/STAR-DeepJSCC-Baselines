"""Semantic-aware DeepJSCC baseline: reconstructs the image AND answers the 7 tasks.

The transmitter is the reconstruction DeepJSCC of djscc_baseline.py (spatial latent, c channels
at 1/d resolution, k = (H/d)(W/d)c/2 complex symbols per image) over the STAR Sionna link.
The receiver decodes the image from the received latent and, from the same received latent,
answers the seven factory tasks with classification heads. Joint loss:

    L = sum_q CE_q  +  w_mse * MSE(image)

So it delivers an image (like STAR and the conventional schemes) plus answers, without any
VLM or LLM. Typically initialised from a trained reconstruction checkpoint (--init).

Usage (from the folder with djscc_baseline.py, task_djscc.py, factory_dataset/, Scene_template/):
  python hybrid_djscc.py train --init djscc_factory_c16.pt --out hybrid_c16.pt | tee hybrid_c16.log
  python hybrid_djscc.py eval  --ckpt hybrid_c16.pt --out hybrid_c16.json
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from djscc_baseline import DeepJSCC, SionnaLink, factory_gt_mask, psnr, num_res  # noqa: E402
from task_djscc import TASKS, Rendered, to_tensor, labels_of, paper_test_set     # noqa: E402

DIGITS = {"pressure": (5, 10, lambda c: ((c + 20) // 10 - 2, (c + 20) % 10)),
          "temperature": (6, 10, lambda c: ((c + 40) // 10 - 4, (c + 40) % 10))}


class TaskHead(nn.Module):
    """Received latent (B, c, h, w) -> 7 task logits."""

    def __init__(self, c, width=64, grid=(6, 10), hidden=1024, digits=True):
        super().__init__()
        w = width
        def blk(ci, co):
            return [nn.Conv2d(ci, co, 3, 2, 1), nn.GroupNorm(8, co), nn.ReLU(inplace=True),
                    nn.Conv2d(co, co, 3, 1, 1), nn.GroupNorm(8, co), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blk(c, w), *blk(w, w), *blk(w, 2 * w), *blk(2 * w, 2 * w),
                                      nn.AdaptiveAvgPool2d(grid))
        self.mlp = nn.Sequential(nn.Linear(2 * w * grid[0] * grid[1], hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        heads = []
        for name, n, _ in TASKS:
            if digits and name in DIGITS:
                d1, d2, f = DIGITS[name]
                heads.append(nn.ModuleList([nn.Linear(hidden, d1), nn.Linear(hidden, d2)]))
                self.register_buffer(f"idx_{name}", torch.tensor([f(c_)[0] * d2 + f(c_)[1] for c_ in range(n)]),
                                     persistent=False)
            else:
                heads.append(nn.Linear(hidden, n))
        self.heads = nn.ModuleList(heads)

    def forward(self, z):
        h = self.mlp(self.features(z).flatten(1))
        out = []
        for (name, n, _), hd in zip(TASKS, self.heads):
            if isinstance(hd, nn.ModuleList):
                a, b = F.log_softmax(hd[0](h), 1), F.log_softmax(hd[1](h), 1)
                out.append((a[:, :, None] + b[:, None, :]).flatten(1)[:, getattr(self, f"idx_{name}")])
            else:
                out.append(hd(h))
        return out


class HybridJSCC(nn.Module):
    def __init__(self, c=16, width=128, down=4, head_width=64, digits=True):
        super().__init__()
        self.jscc = DeepJSCC(c, width, down)
        self.task = TaskHead(c, head_width, digits=digits)
        self.cfg = dict(c=c, width=width, down=down, head_width=head_width, digits=digits)
        self.down, self.c = down, c

    def forward(self, x, link, esn0_db, decode_image=True, frozen_tx=False):
        if frozen_tx:
            with torch.no_grad():
                zc, shape = self.jscc.encode(x)
                zh = link(zc, esn0_db)
        else:
            zc, shape = self.jscc.encode(x)
            zh = link(zc, esn0_db)
        z_real = torch.view_as_real(zh).reshape(shape)            # received latent, (B, c, h, w)
        xh = self.jscc.dec(z_real) if decode_image else None
        return xh, self.task(z_real)


def pad_to(x, d):
    H, W = x.shape[-2:]
    return F.pad(x, (0, (-W) % d, 0, (-H) % d), mode="replicate"), H, W


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = HybridJSCC(**ck["cfg"]).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


def train(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = torch.utils.data.DataLoader(Rendered(a.root, a.seed, a.scale), batch_size=a.batch,
                                     num_workers=a.workers, pin_memory=(dev == "cuda"),
                                     persistent_workers=a.workers > 0)
    model = HybridJSCC(a.c, a.width, a.down, a.head_width, not a.no_digits).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        if "cfg" in ck:                                    # a hybrid checkpoint (e.g. from --freeze_jscc)
            model.load_state_dict(ck["state_dict"]); print("initialised full model from", a.init)
        else:                                              # a reconstruction DeepJSCC checkpoint
            assert ck["config"] == model.jscc.config(), f"--init config {ck['config']} != {model.jscc.config()}"
            model.jscc.load_state_dict(ck["state_dict"]); print("initialised encoder/decoder from", a.init)
    if a.freeze_jscc:
        for p_ in model.jscc.parameters():
            p_.requires_grad_(False)
    link = SionnaLink(device=dev)
    groups = [{"params": model.task.parameters(), "lr": a.lr_task}]
    if not a.freeze_jscc:
        groups.append({"params": model.jscc.parameters(), "lr": a.lr_jscc})
    opt = torch.optim.AdamW(groups, weight_decay=1e-5)
    base = [g["lr"] for g in opt.param_groups]
    start = 1
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optim"])
        start = ck["step"] + 1; print(f"resumed from {a.out} at step {start - 1}")
    warm = max(1, a.steps // 20)
    def scale_at(s):
        return min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    print(f"Hybrid c={a.c} d={a.down} | {'FROZEN DeepJSCC, heads only' if a.freeze_jscc else 'joint'} | w_mse {a.w_mse} | digits {not a.no_digits} | input 1/{a.scale} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params | {dev}", flush=True)

    model.train(); it = iter(dl); t0 = time.time()
    run_ce = run_mse = 0.0; run_c = np.zeros(len(TASKS))
    for step in range(start, a.steps + 1):
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale_at(step - 1)
        x, y = next(it); x, y = x.to(dev, non_blocking=True), y.to(dev)
        xp, H, W = pad_to(x, a.down)
        if a.freeze_jscc:
            model.jscc.eval()
        xh, logits = model(xp, link, random.uniform(*a.snr), decode_image=not a.freeze_jscc,
                           frozen_tx=a.freeze_jscc)
        ce = sum(F.cross_entropy(l, y[:, i]) for i, l in enumerate(logits))
        if a.freeze_jscc:
            mse = torch.zeros((), device=dev); loss = ce
        else:
            mse = F.mse_loss(xh[..., :H, :W], x); loss = ce + a.w_mse * mse
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        run_ce += ce.item(); run_mse += mse.item()
        run_c += np.array([(l.argmax(1) == y[:, i]).float().mean().item() for i, l in enumerate(logits)])
        if step % a.log_every == 0:
            n = a.log_every
            ps = "  -  " if a.freeze_jscc else f"{10*np.log10(1/max(run_mse/n,1e-12)):.2f}"
            print(f"step {step:6d} | ce {run_ce/n:.3f} | psnr {ps} dB | "
                  f"acc {run_c.mean()/n:.3f} | " + " ".join(f"{t[0][:4]} {c/n:.2f}" for t, c in zip(TASKS, run_c)) +
                  f" | {(time.time()-t0)/(step-start+1):.2f} s/step", flush=True)
            run_ce = run_mse = 0.0; run_c = np.zeros(len(TASKS))
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(cfg=model.cfg, state_dict=model.state_dict(), optim=opt.state_dict(),
                            step=step, args=vars(a)), a.out + ".tmp")
            os.replace(a.out + ".tmp", a.out)
    print("saved", a.out)


@torch.no_grad()
def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(a.ckpt, dev); scale = ck["args"]["scale"]
    link = SionnaLink(device=dev)
    test = paper_test_set(a.root)
    H0, W0 = np.asarray(test[0]["img"]).shape[:2]
    res = {"ckpt": a.ckpt, "REs_per_image": num_res(H0 // scale, W0 // scale, model.jscc),
           "input_scale": scale, "tasks": [t[0] for t in TASKS], "snr": {}}
    print(f"{len(test)} test images | {res['REs_per_image']/1e6:.3f} M REs/image")
    torch.manual_seed(0)
    for e in a.snrs:
        correct = np.zeros(len(TASKS)); pf, pt = [], []
        for p in test:
            x = to_tensor(p["img"], scale)[None].to(dev)
            xp, H, W = pad_to(x, model.down)
            xh, logits = model(xp, link, e)
            y = labels_of(p["meta"])
            correct += np.array([int(l.argmax(1).item() == y[i]) for i, l in enumerate(logits)])
            ref = (x[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            rec = (xh[0, :, :H, :W].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            pf.append(psnr(ref, rec)); pt.append(psnr(ref, rec, factory_gt_mask(H, W)))
        per = correct / len(test)
        res["snr"][str(e)] = {"S-Acc": float(per.mean()), "PSNR": float(np.mean(pf)),
                              "S-PSNR_GT": float(np.mean(pt)), **{t[0]: float(v) for t, v in zip(TASKS, per)}}
        print(f"Es/N0 {e:5.1f} dB | S-Acc {per.mean():.3f} | PSNR {np.mean(pf):5.2f} | S-PSNR(GT) {np.mean(pt):5.2f} | "
              + " ".join(f"{t[0][:4]} {v:.2f}" for t, v in zip(TASKS, per)), flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("saved", a.out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="."); t.add_argument("--init", default=None)
    t.add_argument("--c", type=int, default=16); t.add_argument("--width", type=int, default=128)
    t.add_argument("--down", type=int, default=4); t.add_argument("--head_width", type=int, default=64)
    t.add_argument("--freeze_jscc", action="store_true", help="train only the task head on the frozen DeepJSCC")
    t.add_argument("--no_digits", action="store_true"); t.add_argument("--scale", type=int, default=1)
    t.add_argument("--w_mse", type=float, default=1000.0, help="weight of the image MSE vs the summed CE")
    t.add_argument("--batch", type=int, default=2); t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--lr_jscc", type=float, default=5e-5); t.add_argument("--lr_task", type=float, default=1e-3)
    t.add_argument("--snr", type=float, nargs=2, default=[-8.0, 18.0])
    t.add_argument("--workers", type=int, default=2); t.add_argument("--seed", type=int, default=11)
    t.add_argument("--log_every", type=int, default=50); t.add_argument("--save_every", type=int, default=500)
    t.add_argument("--resume", action="store_true"); t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--root", default="."); e.add_argument("--ckpt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--snrs", type=float, nargs="+", default=[-8, -6, -4, -2, 0, 2, 4, 8, 12, 16, 18])
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()

"""Task-oriented DeepJSCC baseline for STAR (no VLM, no LLM).

The transmitter encodes the image into k complex symbols, sent over the STAR Sionna link
(TDL-A, perfect CSI, LMMSE, fixed Es/N0). The receiver does NOT reconstruct the image:
seven classification heads answer the seven factory tasks of Table I directly, and
semantic accuracy (S-Acc) is the classification accuracy. Labels come from the generator.

    status 3 | pressure 41 (2.0-6.0 bar, 0.1) | temperature 56 (40-95 C) | power 2
    room 6   | fire 2 | technician 2

Training images are rendered on the fly with a seed different from the test set
(seed 2027), so training data is unlimited and never contains the test images.

Usage (from the repository root, which contains factory_dataset/ and Scene_template/):
  python djscc/task_djscc.py train --k 256 --out djscc/task_k256.pt
  python djscc/task_djscc.py eval  --ckpt djscc/task_k256.pt --out results/task_k256.json
"""
import argparse, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from djscc_baseline import SionnaLink, AWGNLink  # noqa: E402

# ───────────────────────────── tasks and labels ───────────────────────────────

ROOMS = ["PUMP ROOM", "BOILER ROOM", "COMPRESSOR ROOM", "CHILLER ROOM", "REACTOR ROOM", "UTILITY ROOM"]
TASKS = [  # (name, number of classes, question of Table I)
    ("status", 3, "Is the equipment status NORMAL, WARNING, or FAULT?"),
    ("pressure", 41, "What is the pressure reading?"),
    ("temperature", 56, "What is the temperature reading?"),
    ("power", 2, "Is the power on GRID or BACKUP?"),
    ("room", 6, "Which room is this?"),
    ("fire", 2, "Is there a fire?"),
    ("technician", 2, "In the room, is a maintenance technician PRESENT or ABSENT?"),
]


def labels_of(meta):
    """Generator meta dict -> 7 integer labels (order of TASKS)."""
    return [
        ["NORMAL", "WARNING", "FAULT"].index(meta["equipment_status"]),
        int(round((float(meta["pressure"].split()[0]) - 2.0) / 0.1)),
        int(meta["temp"].split()[0]) - 40,
        ["GRID", "BACKUP"].index(meta["power"]),
        ROOMS.index(meta["room"]),
        ["NO", "YES"].index(meta["fire"]),
        ["NO", "YES"].index(meta["person"]),
    ]


def _gen(root):
    sys.path.insert(0, os.path.join(root, "factory_dataset"))
    import generate_factory_dataset as g
    return g


def to_tensor(img, scale):
    """PIL image -> (3, H/scale, W/scale) float in [0,1]."""
    if scale > 1:
        img = img.resize((img.width // scale, img.height // scale), resample=3)  # bicubic
    return torch.from_numpy(np.asarray(img, np.uint8).copy()).permute(2, 0, 1).float().div(255)


class Rendered(torch.utils.data.IterableDataset):
    """Endless stream of freshly rendered factory images with labels."""

    def __init__(self, root, seed, scale):
        self.root, self.seed, self.scale = root, seed, scale

    def __iter__(self):
        g = _gen(self.root)
        templates = g.load_templates()
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        while True:
            meta = g.make_panel_record(rng)
            img = g.render_panel(meta, templates)
            yield to_tensor(img, self.scale), torch.tensor(labels_of(meta))


# ─────────────────────────────────── model ────────────────────────────────────


class TaskJSCC(nn.Module):
    """CNN encoder -> k complex symbols (unit energy) | MLP receiver -> 7 heads."""

    # digit factorization: class c of a reading -> (first digit, second digit) sub-heads
    DIGITS = {"pressure": (5, 10, lambda c: ((c + 20) // 10 - 2, (c + 20) % 10)),       # 2.0-6.0 bar
              "temperature": (6, 10, lambda c: ((c + 40) // 10 - 4, (c + 40) % 10))}    # 40-95 C

    def __init__(self, k=256, width=64, grid=(6, 10), hidden=1024, digits=False):
        super().__init__()
        w = width
        def blk(ci, co):
            return [nn.Conv2d(ci, co, 3, 2, 1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                    nn.Conv2d(co, co, 3, 1, 1), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*blk(3, w), *blk(w, w), *blk(w, 2 * w), *blk(2 * w, 2 * w),
                                      *blk(2 * w, 4 * w), nn.AdaptiveAvgPool2d(grid))
        self.to_sym = nn.Linear(4 * w * grid[0] * grid[1], 2 * k)
        self.rx = nn.Sequential(nn.Linear(2 * k, hidden), nn.ReLU(inplace=True),
                                nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
        self.digits = digits
        heads = []
        for name, n, _ in TASKS:
            if digits and name in self.DIGITS:
                d1, d2, f = self.DIGITS[name]
                heads.append(nn.ModuleList([nn.Linear(hidden, d1), nn.Linear(hidden, d2)]))
                self.register_buffer(f"idx_{name}", torch.tensor([f(c)[0] * d2 + f(c)[1] for c in range(n)]),
                                     persistent=False)
            else:
                heads.append(nn.Linear(hidden, n))
        self.heads = nn.ModuleList(heads)
        self.k, self.cfg = k, dict(k=k, width=width, grid=list(grid), hidden=hidden, digits=digits)

    def encode(self, x):
        z = self.to_sym(self.features(x).flatten(1))
        zc = torch.view_as_complex(z.reshape(z.shape[0], self.k, 2).contiguous())
        return zc / torch.sqrt(zc.abs().pow(2).mean(1, keepdim=True) + 1e-12)   # Es = 1

    def decode(self, zh):
        h = self.rx(torch.view_as_real(zh).reshape(zh.shape[0], -1))
        out = []
        for (name, n, _), hd in zip(TASKS, self.heads):
            if isinstance(hd, nn.ModuleList):        # log p(c) = log p(d1) + log p(d2), restricted to valid c
                a, b = F.log_softmax(hd[0](h), 1), F.log_softmax(hd[1](h), 1)
                joint = (a[:, :, None] + b[:, None, :]).flatten(1)
                out.append(joint[:, getattr(self, f"idx_{name}")])
            else:
                out.append(hd(h))
        return out


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"]); cfg["grid"] = tuple(cfg["grid"])
    m = TaskJSCC(**cfg).to(device); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


# ─────────────────────────────────── train ────────────────────────────────────


def train(a):
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = Rendered(a.root, a.seed, a.scale)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers,
                                     pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0)
    model = TaskJSCC(a.k, a.width, digits=a.digits).to(dev)
    link = SionnaLink(device=dev) if a.channel == "sionna" else AWGNLink()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = max(1, a.steps // 20)                       # linear warm-up, then cosine decay

    def lr_at(s):                                       # s = completed steps
        return a.lr * min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))

    start = 1
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["optim"])
        start = ck["step"] + 1; print(f"resumed from {a.out} at step {start - 1}")
    print(f"TaskJSCC k={a.k} REs/image | width {a.width} | input 1/{a.scale} | digits {a.digits} | "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params | {a.channel} | {dev}", flush=True)

    model.train(); it = iter(dl); t0 = time.time(); run_l, run_c = 0.0, np.zeros(len(TASKS))
    for step in range(start, a.steps + 1):
        x, y = next(it); x, y = x.to(dev, non_blocking=True), y.to(dev)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step - 1)
        snr = random.uniform(*a.snr)
        logits = model.decode(link(model.encode(x), snr))
        loss = sum(F.cross_entropy(l, y[:, i]) for i, l in enumerate(logits))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        run_l += loss.item()
        run_c += np.array([(l.argmax(1) == y[:, i]).float().mean().item() for i, l in enumerate(logits)])
        if step % a.log_every == 0:
            n = a.log_every
            acc = " ".join(f"{t[0][:4]} {c/n:.2f}" for t, c in zip(TASKS, run_c))
            print(f"step {step:6d} | loss {run_l/n:.3f} | acc {run_c.mean()/n:.3f} | {acc} | "
                  f"{(time.time()-t0)/(step-start+1):.2f} s/step", flush=True)
            run_l, run_c = 0.0, np.zeros(len(TASKS))
        if step % a.save_every == 0 or step == a.steps:
            torch.save(dict(cfg=model.cfg, state_dict=model.state_dict(), optim=opt.state_dict(),
                            step=step, args=vars(a)), a.out + ".tmp")
            os.replace(a.out + ".tmp", a.out)
    print("saved", a.out)


# ──────────────────────────────────── eval ────────────────────────────────────


def paper_test_set(root, n_per_template=20, seed_run=20240612, n_panels=300):
    """The 60 test images of the paper (same sampling as the factory notebook, Section 6)."""
    g = _gen(root)
    recs = g.load_dataset(n_panels=n_panels)
    rng = random.Random(seed_run); out = []
    for t in ["fire", "human", "clear"]:
        cands = [p for p in recs if p["meta"]["template"] == t]
        out += rng.sample(cands, min(n_per_template, len(cands)))
    rng.shuffle(out)
    return out


@torch.no_grad()
def evaluate(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load(a.ckpt, dev); scale = ck["args"]["scale"]
    link = SionnaLink(device=dev)
    test = paper_test_set(a.root)
    X = torch.stack([to_tensor(p["img"], scale) for p in test])
    Y = torch.tensor([labels_of(p["meta"]) for p in test])
    torch.manual_seed(0)
    res = {"k_REs_per_image": model.k, "ckpt": a.ckpt, "n_images": len(test), "repeats": a.repeats,
           "tasks": [t[0] for t in TASKS], "snr": {}}
    print(f"{len(test)} test images | k = {model.k} REs/image")
    for e in a.snrs:
        correct = np.zeros(len(TASKS)); n = 0
        for _ in range(a.repeats):                       # independent channel draws
            for s in range(0, len(X), a.batch):
                x, y = X[s:s + a.batch].to(dev), Y[s:s + a.batch].to(dev)
                logits = model.decode(link(model.encode(x), e))
                correct += np.array([(l.argmax(1) == y[:, i]).sum().item() for i, l in enumerate(logits)])
                n += len(x)
        per = correct / n
        res["snr"][str(e)] = {"S-Acc": float(per.mean()), **{t[0]: float(p) for t, p in zip(TASKS, per)}}
        print(f"Es/N0 {e:5.1f} dB | S-Acc {per.mean():.3f} | " +
              " ".join(f"{t[0][:4]} {p:.2f}" for t, p in zip(TASKS, per)), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("saved", a.out)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", default="."); t.add_argument("--k", type=int, default=256)
    t.add_argument("--width", type=int, default=64); t.add_argument("--scale", type=int, default=2)
    t.add_argument("--batch", type=int, default=16); t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--lr", type=float, default=1e-3); t.add_argument("--snr", type=float, nargs=2, default=[-8.0, 18.0])
    t.add_argument("--channel", choices=["sionna", "awgn"], default="sionna")
    t.add_argument("--workers", type=int, default=2); t.add_argument("--seed", type=int, default=7)
    t.add_argument("--log_every", type=int, default=100); t.add_argument("--save_every", type=int, default=1000)
    t.add_argument("--digits", action="store_true", help="read pressure/temperature digit by digit")
    t.add_argument("--resume", action="store_true"); t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--root", default="."); e.add_argument("--ckpt", required=True); e.add_argument("--out", required=True)
    e.add_argument("--snrs", type=float, nargs="+", default=[-8, -6, -4, -2, 0, 2, 4, 8, 12, 16, 18])
    e.add_argument("--repeats", type=int, default=5); e.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()

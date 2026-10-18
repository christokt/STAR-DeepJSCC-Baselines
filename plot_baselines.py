"""Paper-style curves: S-Acc and S-PSNR vs Es/N0 for the DeepJSCC baselines (and STAR).

Each --run is LABEL=path/to/result.json, as written by `task_djscc.py eval` or
`hybrid_djscc.py eval`. Optional --star adds STAR (and the conventional scheme) from a
`base_runs_final.pkl` of the factory notebook (Section 6), for one base modulation.

  python plot_baselines.py --run "Task JSCC (k=256)=task_k256.json" \
                           --run "Semantic DeepJSCC=hybrid_c16.json" \
                           --star results/factory/STAR_DS4_conv_DS4/base_runs_final.pkl --star_qam 256qam \
                           --out figs/baselines

Writes <out>_sacc.pdf and <out>_spsnr.pdf (3.5 in wide, IEEE one-column) and PNG previews.
S-PSNR is drawn only for runs that deliver an image. NOTE: the S-PSNR stored in
base_runs_final.pkl is computed on STAR's own SCI mask, not on the ground-truth region.
"""
import argparse, json, os, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#7C3AED", "#EA580C", "#0891B2"]
MARKERS = ["s", "^", "D", "v", "P", "X"]


def style():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150})


def ax_style(ax):
    ax.grid(True, linestyle="--", color="#D8DEE9", linewidth=0.7); ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def load_run(path):
    d = json.load(open(path))
    snr = sorted(d["snr"], key=float)
    x = [float(s) for s in snr]
    acc = [d["snr"][s]["S-Acc"] for s in snr]
    sp = [d["snr"][s]["S-PSNR_GT"] for s in snr] if "S-PSNR_GT" in d["snr"][snr[0]] else None
    res = d.get("REs_per_image", d.get("k_REs_per_image"))
    return x, acc, sp, res


def fmt_res(n):
    return f"{n/1e6:.2f} M REs" if n >= 1e5 else f"{n:,} REs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help='"LABEL=path.json" (repeatable)')
    ap.add_argument("--star", default=None, help="base_runs_final.pkl from the factory notebook")
    ap.add_argument("--star_qam", default="256qam")
    ap.add_argument("--star_label", default="STAR (256-QAM, D=4)")
    ap.add_argument("--conv_label", default=None, help="label for the conventional scheme in the pkl (omit to hide)")
    ap.add_argument("--out", default="baselines")
    a = ap.parse_args()
    style()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    f1, ax1 = plt.subplots(figsize=(3.5, 2.6)); f2, ax2 = plt.subplots(figsize=(3.5, 2.6))
    drew_sp = False

    if a.star:
        r = pickle.load(open(a.star, "rb"))[a.star_qam]["res"]      # (snr, sci_acc, base_acc, sci_sp, base_sp)
        x = [t[0] for t in r]
        ax1.plot(x, [t[1] for t in r], "o-", color="#F5B700", lw=1.6, ms=3.5, label=a.star_label, zorder=4)
        ax2.plot(x, [t[3] for t in r], "o-", color="#F5B700", lw=1.6, ms=3.5, label=a.star_label, zorder=4)
        drew_sp = True
        if a.conv_label:
            ax1.plot(x, [t[2] for t in r], "o--", color="#6B7280", lw=1.2, ms=3, label=a.conv_label)
            ax2.plot(x, [t[4] for t in r], "o--", color="#6B7280", lw=1.2, ms=3, label=a.conv_label)

    for i, spec in enumerate(a.run):
        label, path = spec.rsplit("=", 1)       # last "=" separates the path; labels may contain "="
        x, acc, sp, res = load_run(path)
        lab = f"{label} ({fmt_res(res)})" if res else label
        c, m = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        ax1.plot(x, acc, m + "-", color=c, lw=1.3, ms=3.2, label=lab)
        if sp is not None:
            ax2.plot(x, sp, m + "-", color=c, lw=1.3, ms=3.2, label=lab); drew_sp = True

    ax1.set_xlabel(r"$E_s/N_0$ (dB)"); ax1.set_ylabel("Semantic accuracy"); ax1.set_ylim(0, 1.03)
    ax_style(ax1); ax1.legend(loc="lower right", framealpha=0.9)
    f1.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        f1.savefig(f"{a.out}_sacc.{ext}", bbox_inches="tight")
    print("saved", f"{a.out}_sacc.pdf")
    if drew_sp:
        ax2.set_xlabel(r"$E_s/N_0$ (dB)"); ax2.set_ylabel("S-PSNR (dB)")
        ax_style(ax2); ax2.legend(loc="upper left", framealpha=0.9)
        f2.tight_layout(pad=0.5)
        for ext in ("pdf", "png"):
            f2.savefig(f"{a.out}_spsnr.{ext}", bbox_inches="tight")
        print("saved", f"{a.out}_spsnr.pdf")


if __name__ == "__main__":
    main()

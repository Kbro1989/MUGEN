"""Visualize Weighted-HS router decisions by action group (CPU-only).

Consumes the npz/meta.json written by scripts/analyze_router_decision.py and
answers "which layers does each action activate?": global K x L heatmap,
per-action mean and difference (action mean - global mean) heatmaps, per-slot
routing entropy, action-vs-action Jensen-Shannon divergence with hierarchical
clustering, and a between/within variance ratio per slot (the statistical
test of "does the router decide BY ACTION?"). Also writes summary.md with a
top-3-layers-per-action table.

Usage (login node is fine, no GPU):

    python scripts/plot_router_decision.py <analysis_dir> [<analysis_dir2> ...]

Each <analysis_dir> must contain router_decisions.npz + meta.json; figures
and summary.md are written into the same dir. Passing several dirs (e.g.
epoch snapshots) additionally writes evolution_global_mean.png comparing
their global mean routing side by side.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ACTIONS = ['walk', 'run', 'jump', 'dance', 'sit', 'stand', 'turn', 'kick',
           'throw', 'wave', 'pick', 'crawl', 'climb', 'crouch', 'squat',
           'punch', 'stretch', 'bend', 'swim', 'balance', 'spin', 'clap']
MIN_GROUP = int(os.environ.get('PLOT_MIN_GROUP', 30))


def action_regex(verb):
    stem = verb[:-1] if verb.endswith('e') else verb
    forms = {verb, verb + 's', verb + 'ed', verb + 'es', stem + 'ing'}
    if verb in ('run', 'sit', 'clap', 'spin'):  # CVC doubling
        forms.add(verb + verb[-1] + 'ing')
        forms.add(verb + verb[-1] + 'ed')
    return re.compile(r'\b(' + '|'.join(sorted(forms)) + r')\b', re.I)


def entropy(p, axis=-1):
    q = np.clip(p, 1e-12, None)
    return -(q * np.log(q)).sum(axis=axis)


def js_div(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (entropy(m) - 0.5 * entropy(p) - 0.5 * entropy(q)) * 2  # in nats


def heat(ax, M, title, vmin=None, vmax=None, cmap='viridis'):
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('GPT-2 layer', fontsize=8)
    ax.set_ylabel('slot k', fontsize=8)
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.tick_params(labelsize=7)
    return im


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def analyze_dir(d):
    z = np.load(os.path.join(d, 'router_decisions.npz'))
    meta = json.load(open(os.path.join(d, 'meta.json')))
    W, G, texts = z['weights'], z['logits'], meta['texts']   # (N,K,L)
    N, K, L = W.shape
    tag = f"{meta.get('split','?')} ep{meta.get('epoch','?')} tau={float(z['tau']):.2f}"
    gmean = W.mean(axis=0)                        # (K,L)

    # group texts by action verb (a text may join several groups)
    groups = {}
    for a in ACTIONS:
        rx = action_regex(a)
        idx = np.array([i for i, t in enumerate(texts) if rx.search(t)])
        if len(idx) >= MIN_GROUP:
            groups[a] = idx
    acts = sorted(groups, key=lambda a: -len(groups[a]))
    amean = {a: W[groups[a]].mean(axis=0) for a in acts}   # (K,L) each

    # -- fig 1: global mean --
    fig, ax = plt.subplots(figsize=(6, 3))
    im = heat(ax, gmean, f'Global mean routing weights ({tag}, N={N})')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'fig1_global_mean.png'), dpi=150)
    plt.close(fig)

    # -- fig 2/3: per-action mean and diff heatmap grids --
    if acts:
        for figname, mats, cmap, sym, subtitle in (
                ('fig2_action_mean.png', amean, 'viridis', False, 'mean weights'),
                ('fig3_action_diff.png',
                 {a: amean[a] - gmean for a in acts}, 'coolwarm', True,
                 'action mean - global mean')):
            ncol = 4
            nrow = (len(acts) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol,
                                     figsize=(4 * ncol, 2.2 * nrow), squeeze=False)
            vmax = max(np.abs(m).max() for m in mats.values())
            vmin = -vmax if sym else 0.0
            for i, a in enumerate(acts):
                ax = axes[i // ncol][i % ncol]
                im = heat(ax, mats[a], f'{a} (n={len(groups[a])})',
                          vmin=vmin, vmax=vmax, cmap=cmap)
            for j in range(len(acts), nrow * ncol):
                axes[j // ncol][j % ncol].axis('off')
            fig.suptitle(f'Per-action {subtitle} ({tag})', fontsize=11)
            fig.colorbar(im, ax=[ax for row in axes for ax in row],
                         shrink=0.6, pad=0.01)
            fig.savefig(os.path.join(d, figname), dpi=150, bbox_inches='tight')
            plt.close(fig)

    # -- fig 4: per-slot entropy --
    sample_ent = entropy(W).mean(axis=0)          # (K,) mean per-sample entropy
    mean_ent = entropy(gmean)                     # (K,) entropy of the mean
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(K)
    ax.bar(x - 0.2, sample_ent, 0.4, label='mean per-sample entropy')
    ax.bar(x + 0.2, mean_ent, 0.4, label='entropy of mean routing')
    ax.axhline(np.log(L), ls='--', c='gray', lw=1, label=f'uniform ln{L}')
    ax.set_xlabel('slot k')
    ax.set_ylabel('entropy (nats)')
    ax.set_title(f'Routing entropy per slot ({tag})', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'fig4_slot_entropy.png'), dpi=150)
    plt.close(fig)

    # -- fig 5: JS divergence between actions + clustering --
    lines = [f'# Router decision summary — {tag}', '',
             f'- N={N}, K={K}, L={L}; action groups (>= {MIN_GROUP} texts): '
             + (', '.join(f'{a}({len(groups[a])})' for a in acts) or 'NONE'), '']
    if len(acts) >= 2:
        n = len(acts)
        JS = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                JS[i, j] = np.mean([js_div(amean[acts[i]][k], amean[acts[j]][k])
                                    for k in range(K)])
        order = list(range(n))
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            order = list(leaves_list(linkage(
                squareform(0.5 * (JS + JS.T), checks=False), method='average')))
        except Exception as e:
            lines.append(f'(scipy unavailable, unclustered JS order: {e})')
        JS_o = JS[np.ix_(order, order)]
        labels = [acts[i] for i in order]
        fig, ax = plt.subplots(figsize=(0.45 * n + 2, 0.45 * n + 1.5))
        im = ax.imshow(JS_o, cmap='magma')
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f'Action-action routing JS divergence ({tag})', fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(d, 'fig5_js_matrix.png'), dpi=150)
        plt.close(fig)
        iu = np.triu_indices(n, 1)
        lines += [f'- JS divergence (nats): mean {JS[iu].mean():.4f}, '
                  f'max {JS[iu].max():.4f} '
                  f'({acts[int(iu[0][JS[iu].argmax()])]} vs {acts[int(iu[1][JS[iu].argmax()])]})',
                  '']

    # -- between/within variance ratio per slot --
    if len(acts) >= 2:
        ratios = []
        for k in range(K):
            gm = np.stack([amean[a][k] for a in acts])          # (A,L)
            between = gm.var(axis=0).mean()
            within = np.mean([W[groups[a]][:, k, :].var(axis=0).mean() for a in acts])
            ratios.append(between / max(within, 1e-12))
        lines += ['- Between/within variance ratio per slot (router decides '
                  'BY ACTION if >> 0): '
                  + ', '.join(f'k{k}={r:.3f}' for k, r in enumerate(ratios)), '']

    # -- saturation diagnosis + logit-space analysis (weights can be one-hot
    # while the interesting structure, if any, lives in the raw logits) --
    srt = np.sort(G, axis=-1)
    margin = (srt[..., -1] - srt[..., -2]).mean(axis=0)        # (K,) top1-top2 gap
    mode = G.mean(axis=0).argmax(axis=-1)                      # (K,) global mode layer
    mode_frac = (G.argmax(axis=-1) == mode[None, :]).mean(axis=0)
    lines += ['## Saturation / static-routing diagnosis',
              '- mean top1-top2 logit margin per slot: '
              + ', '.join(f'k{k}={m:.2f}' for k, m in enumerate(margin)),
              '- fraction of texts whose argmax layer == global mode: '
              + ', '.join(f'k{k}={f:.3f}(L{mode[k]})' for k, f in enumerate(mode_frac)),
              '- logit std across texts: max '
              f'{G.std(axis=0).max():.3f}, mean {G.std(axis=0).mean():.3f}',
              '- verdict: routing is '
              + ('STATIC per slot (text-independent; margins dwarf text-driven '
                 'variation)' if (margin > 5 * G.std(axis=0).mean(axis=-1)).all()
                 and (mode_frac > 0.99).all() else 'text-dependent (see figs)'),
              '']
    if acts:
        aG = {a: G[groups[a]].mean(axis=0) - G.mean(axis=0) for a in acts}
        ncol = 4
        nrow = (len(acts) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol,
                                 figsize=(4 * ncol, 2.2 * nrow), squeeze=False)
        vmax = max(np.abs(m).max() for m in aG.values())
        for i, a in enumerate(acts):
            ax = axes[i // ncol][i % ncol]
            im = heat(ax, aG[a], f'{a} (n={len(groups[a])})',
                      vmin=-vmax, vmax=vmax, cmap='coolwarm')
        for j in range(len(acts), nrow * ncol):
            axes[j // ncol][j % ncol].axis('off')
        fig.suptitle(f'Per-action LOGIT diff, action mean - global mean ({tag})',
                     fontsize=11)
        fig.colorbar(im, ax=[ax for row in axes for ax in row], shrink=0.6, pad=0.01)
        fig.savefig(os.path.join(d, 'fig6_action_logit_diff.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        # JS between actions in a softened view (T=2) of the logits: does ANY
        # per-action structure exist below the saturated weights?
        SW = {a: softmax(G[groups[a]] / 2.0).mean(axis=0) for a in acts}
        n = len(acts)
        js2 = np.array([[np.mean([js_div(SW[a][k], SW[b][k]) for k in range(K)])
                         for b in acts] for a in acts])
        iu = np.triu_indices(n, 1)
        lines += [f'- softened (T=2) action JS: mean {js2[iu].mean():.5f}, '
                  f'max {js2[iu].max():.5f} — sub-saturation per-action structure '
                  'exists if these are clearly > 0',
                  '- max |action logit diff|: '
                  f'{max(np.abs(m).max() for m in aG.values()):.3f} '
                  '(vs inter-layer margins above)', '']

    # -- top-3 layers per action (the "action -> layers a,b,c" table) --
    lines += ['## Top-3 layers per action (weights summed over slots)', '',
              '| action | n | top layers (weight share) |', '|---|---|---|']
    pooled_g = gmean.mean(axis=0)
    top_g = np.argsort(-pooled_g)[:3]
    lines.append('| (global) | %d | %s |' % (
        N, ', '.join(f'L{l} ({pooled_g[l]:.2f})' for l in top_g)))
    for a in acts:
        pooled = amean[a].mean(axis=0)                          # (L,)
        top = np.argsort(-pooled)[:3]
        lines.append('| %s | %d | %s |' % (
            a, len(groups[a]), ', '.join(f'L{l} ({pooled[l]:.2f})' for l in top)))
    lines += ['', '## Per-slot entropy',
              '- mean per-sample: ' + ', '.join(f'k{k}={v:.3f}' for k, v in enumerate(sample_ent)),
              '- of mean routing: ' + ', '.join(f'k{k}={v:.3f}' for k, v in enumerate(mean_ent)),
              f'- uniform reference ln(L) = {np.log(L):.3f}', '']
    with open(os.path.join(d, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f'[{d}] figures + summary.md written; groups={len(acts)}')
    return gmean, tag


def main():
    dirs = sys.argv[1:]
    assert dirs, 'usage: plot_router_decision.py <analysis_dir> [...]'
    results = [analyze_dir(d) for d in dirs]
    if len(results) > 1:
        fig, axes = plt.subplots(1, len(results),
                                 figsize=(5 * len(results), 3), squeeze=False)
        vmax = max(g.max() for g, _ in results)
        for ax, (g, tag) in zip(axes[0], results):
            im = heat(ax, g, tag, vmin=0, vmax=vmax)
        fig.suptitle('Global mean routing across checkpoints', fontsize=11)
        fig.colorbar(im, ax=list(axes[0]), shrink=0.7, pad=0.01)
        out = os.path.join(dirs[0], 'evolution_global_mean.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print('wrote', out)


if __name__ == '__main__':
    main()

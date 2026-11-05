import matplotlib.pyplot as plt
import numpy as np
from qutip import bell_state, ket2dm, Qobj

from ..channel import (
    AtmModel, jitter_vis, loss_ch, phase_noise, pol_drift, transmittance,
)
from ..config import MAX_MOBILE_KM, SPEED_OF_LIGHT
from .style import (
    ax_style, BLUE, DARK, GRAY, GREEN, ORANGE, PURPLE, RED, save_fig, TEAL,
)
from ..quantum_metrics import metrics


def fig_noise_breakdown():
    dists = np.linspace(0.3, MAX_MOBILE_KM, 40)
    atm = AtmModel(); V = jitter_vis()

    def sim(d, use_atm=True, use_pol=True, use_ph=True,
            use_loss=True, use_jit=True, n=40):
        phi0 = ket2dm(bell_state("00")); Fs=[]
        for _ in range(n):
            rho = phi0.copy()
            transit = d*1e3/SPEED_OF_LIGHT
            if use_atm: rho = atm.decohere(rho, transit)
            if use_pol: rho = pol_drift(rho, d)
            if use_ph:  rho = phase_noise(rho, d)
            T = transmittance(d) if use_loss else 1.0
            rho = loss_ch(rho, T)
            if use_jit:
                rn=rho.full(); mask=1-np.eye(4)
                rn=rn*(1-mask*(1-V)); rho=Qobj(rn,dims=[[2,2],[2,2]])
            tr=rho.tr().real
            if tr>0: rho=rho/tr
            F,*_=metrics(rho); Fs.append(F)
        return np.mean(Fs), np.std(Fs)/np.sqrt(n)

    print("  Computing noise breakdown ...")
    configs = [
        ('All noise sources',      dict(),                                             RED,    '-',  2.5),
        ('w/o Atmospheric decay',  dict(use_atm=False),                               BLUE,   '--', 1.8),
        ('w/o Polarization drift', dict(use_pol=False),                               GREEN,  '--', 1.8),
        ('w/o Phase noise',        dict(use_ph=False),                                ORANGE, '--', 1.8),
        ('w/o Channel loss',       dict(use_loss=False),                              PURPLE, '--', 1.8),
        ('w/o Timing jitter',      dict(use_jit=False),                               TEAL,   '--', 1.8),
        ('Ideal (no noise)',        dict(use_atm=False,use_pol=False,use_ph=False,
                                        use_loss=False,use_jit=False),                 GRAY,   ':',  1.5),
    ]
    curves = {}
    for label, kw, *_ in configs:
        ms, es = zip(*[sim(d, **kw) for d in dists])
        curves[label] = (np.array(ms), np.array(es))

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,7))
    fig.patch.set_facecolor('white')
    for ax in (ax1,ax2): ax_style(ax)

    for (label, _, color, ls, lw) in configs:
        mu, err = curves[label]
        ax1.plot(dists, mu, color=color, ls=ls, lw=lw, label=label, zorder=3)
        if ls=='-': ax1.fill_between(dists, mu-err, mu+err, color=color, alpha=0.15)
    ax1.axhline(0.5, color=RED,   ls=':', lw=1.2, alpha=0.6, label='Ent. threshold')
    ax1.axhline(2/3, color=GREEN, ls=':', lw=1.2, alpha=0.6, label='QKD threshold')
    ax1.set_xlabel('Mobile Link Distance (km)', fontsize=10, color=DARK)
    ax1.set_ylabel('Bell State Fidelity F(Φ⁺)', fontsize=10, color=DARK)
    ax1.set_title('(A)  Fidelity vs Distance -- Noise Sources', fontsize=11,
                  fontweight='bold', color=DARK, pad=7)
    ax1.legend(fontsize=7.5, loc='upper right')
    ax1.set_ylim(0.2,1.02); ax1.set_xlim(0, MAX_MOBILE_KM)

    mu_all = curves['All noise sources'][0]
    contrib_names = [
        ('Atmospheric decay',  'w/o Atmospheric decay'),
        ('Polarization drift', 'w/o Polarization drift'),
        ('Phase noise',        'w/o Phase noise'),
        ('Channel loss',       'w/o Channel loss'),
        ('Timing jitter',      'w/o Timing jitter'),
    ]
    contrib_colors = [BLUE, GREEN, ORANGE, PURPLE, TEAL]
    d_idx = np.argmin(np.abs(dists - 2.5))
    bars = [curves[wkey][0][d_idx] - mu_all[d_idx] for _, wkey in contrib_names]
    blabs = [n for n,_ in contrib_names]

    ax2.bar(range(len(bars)), bars, color=contrib_colors, edgecolor=DARK,
            lw=0.8, alpha=0.88, width=0.65, zorder=3)
    for i,(val,lab) in enumerate(zip(bars,blabs)):
        ax2.text(i, val+0.001, f'+{val:.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=DARK)
    ax2.set_xticks(range(len(blabs)))
    ax2.set_xticklabels(blabs, fontsize=9, color=DARK)
    ax2.set_ylabel('Fidelity improvement ΔF  (at 2.5 km)', fontsize=10, color=DARK)
    ax2.set_title('(B)  Marginal Noise Contribution (2.5 km)', fontsize=11,
                  fontweight='bold', color=DARK, pad=7)

    fig.suptitle('Noise Breakdown -- Mobile FSO Quantum Links', fontsize=13,
                 fontweight='bold', color=DARK)
    plt.tight_layout()
    return save_fig(fig, "fig08_noise_breakdown.png")

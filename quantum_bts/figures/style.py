from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from ..config import OUT


BLUE   = "#1A6DB5"
TEAL   = "#0E9E8E"
ORANGE = "#E07B1A"
RED    = "#D63A2F"
GREEN  = "#2E9E4F"
PURPLE = "#6A3FA8"
GRAY   = "#6B7280"
DARK   = "#1F2937"
PINK   = "#E91E8C"
RADIO_COLORS = {"GSM": "#E07B1A", "UMTS": "#1A6DB5", "LTE": "#2E9E4F"}
cmap_fid = LinearSegmentedColormap.from_list("fid", ["#D63A2F","#F5A623","#2E9E4F"], N=256)


def ax_style(ax, title=None):
    ax.set_facecolor('#FAFBFC')
    ax.grid(True, linestyle=':', color='#CBD5E1', alpha=0.7, zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor('#CBD5E1')
    ax.tick_params(labelsize=8.5, color=DARK)
    if title: ax.set_title(title, fontsize=11, fontweight='bold', color=DARK, pad=7)


def save_fig(fig, name):
    path = f"{OUT}/{name}"
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[✓] {path}")
    return path


MOB_TYPE_COLORS  = {'pedestrian': GREEN,  'vehicle': ORANGE, 'IoT': PURPLE}
MOB_TYPE_MARKERS = {'pedestrian': 'o',    'vehicle': 's',    'IoT': '^'}

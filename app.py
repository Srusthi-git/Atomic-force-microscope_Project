
from argparse import ArgumentParser
import pickle

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@st.cache_data
def load_data(prefix):
    print(f"- Loading {prefix}{{.data.pickled,.heights.npy}}")
    hname = f"{prefix}.heights.npy"
    H = np.load(hname)
    m, n = H.shape
    fname = f"{prefix}.data.pickled"
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    # Estimate all slopes
    slope_est = dict()  # maps tuple (s, i, j) to tuple (slope, anchors, info)
    for point, curve in data.items():
        s, i, j = point
        slope_est[point] = estimate_slope(curve, s)
    nseries = 1 + max(s for s, i, j in data.keys())
    slope_heatmaps = []
    for s in range(nseries):
        M = np.array([[(slope_est[s, i, j][0]) for j in range(n)] for i in range(m)], dtype=np.float64)
        slope_heatmaps.append(M)
    return H, data, slope_est, slope_heatmaps


def do_plot(point, curve, slope=None, anchors=None):
    """plot one distance-force curve with estimated slope"""
    s, i, j = point
    d, f = curve
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel("distance (m)")
    plt.ylabel("force (N)")
    mode = 'push' if s == 0 else 'retract'
    plt.title(f"{mode} at ({i}, {j});  number of records: {len(d)}")
    label = f'data: {mode} at {(i, j)}'
    plt.scatter(d, f, s=1, label=label)
    plt.grid()
    if slope is not None and anchors is not None:
        anchor0, anchor1 = anchors[0], anchors[1]
        plt.axline(anchor0, slope=slope, color='red', linestyle='--', label=f'{slope:.4g} N/m')
        plt.plot([anchor0[0]], [anchor0[1]], 'rx') 
        plt.plot([anchor1[0]], [anchor1[1]], 'rx')
    plt.legend()
    return fig


def estimate_slope(curve, s, nan=float("nan")):
    d, f = curve
    if s == 0:
        d, f = d[::-1], f[::-1]  

    
    mask = f <= 1.2e-09
    d_process = d[mask]
    f_process = f[mask]

    window = int(len(f_process) / 15)

    d_windows = np.lib.stride_tricks.sliding_window_view(d_process, window)
    f_windows = np.lib.stride_tricks.sliding_window_view(f_process, window)

    
    x = d_windows - d_windows.mean(axis=1, keepdims=True)
    y = f_windows - f_windows.mean(axis=1, keepdims=True)

    num = np.sum(x * y, axis=1)
    deno = np.sum(x ** 2, axis=1)
    f_squared_diff = np.sum(y ** 2, axis=1)

    slopes = num / deno
    r_squared = (num ** 2) / (deno * f_squared_diff)

    best_index = np.argmax(r_squared)
    slope = slopes[best_index]
    best_d = d_windows[best_index]
    best_f = f_windows[best_index]
    anchor1 = (best_d[0], best_f[0])
    anchor2 = (best_d[-1], best_f[-1])
    anchors = (anchor1, anchor2)
    info = None  
    return (slope, anchors, info)

p = ArgumentParser()
p.add_argument("prefix", help="common path prefix for spectra (.data.pickled) and heights (.heights.npy)")
args = p.parse_args()
prefix = args.prefix

st.sidebar.title("AFM Data Explorer")
st.sidebar.write(f"Path prefix:\n'{prefix}'")
H, S, slope_est, slope_heatmaps = load_data(prefix)  
m, n = H.shape
nseries = len(slope_heatmaps)

s = st.sidebar.radio("Series:", ["0[push]", "1[retract]"])
s = int(s.replace("[push]", "").replace("[retract]", ""))
i = st.sidebar.slider("Coordinate i (vertical)", 0, m-1, 0)
j = st.sidebar.slider("Coordinate j (horizontal)", 0, n-1, 0)

st.header("Heights")
plt.figure(figsize=(10, 6))
plt.imshow(H, cmap='turbo', aspect='auto')
a=plt.colorbar(label='Height')
a.set_ticks([0, 50e-9, 100e-9, 150e-9, 200e-9,250e-9])
a.set_ticklabels(["0", "50n", "100n", "150n", "200n", "250n"])
st.pyplot(plt)

st.header(f"Slopes")
plt.figure(figsize=(10, 6))
plt.imshow(slope_heatmaps[s], cmap='turbo', aspect='auto')
plt.xlim()
plt.ylim(120, 0)
plt.colorbar(label='Slope (N/m)')
st.pyplot(plt)

point = (s, i, j)
curve = S[point]
slope, anchors, _ = slope_est[point]
fig = do_plot(point, curve, slope, anchors)
st.pyplot(fig)



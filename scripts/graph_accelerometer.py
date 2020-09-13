#!/usr/bin/env python2
# Generate adxl345 accelerometer graphs
#
# Copyright (C) 2020  Kevin O'Connor <kevin@koconnor.net>
# Copyright (C) 2020  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import optparse
import numpy as np, matplotlib

def parse_log(logname):
    f = open(logname, 'r')
    out = []
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) != 4:
            continue
        try:
            fparts = [float(p) for p in parts]
        except ValueError:
            continue
        out.append(fparts)
    return out


######################################################################
# Raw accelerometer graphing
######################################################################

def plot_accel(data, logname):
    first_time = data[0][0]
    times = [d[0] - first_time for d in data]
    fig, axes = matplotlib.pyplot.subplots(nrows=3, sharex=True)
    axes[0].set_title("Accelerometer data (%s)" % (logname,))
    axis_names = ['x', 'y', 'z']
    for i in range(len(axis_names)):
        adata = [d[i+1] for d in data]
        avg = sum(adata) / len(adata)
        adata = [ad - avg for ad in adata]
        ax = axes[i]
        ax.plot(times, adata, alpha=0.8)
        ax.grid(True)
        ax.set_ylabel('%s accel (%+.3f)\n(mm/s^2)' % (axis_names[i], -avg))
    axes[-1].set_xlabel('Time (%+.3f)\n(s)' % (-first_time,))
    fig.tight_layout()
    return fig


######################################################################
# Frequency graphing
######################################################################

MAX_FREQ = 200.
WINDOW_T_SEC = 0.5

class CalibrationData:
    def __init__(self, freq_bins, psd_sum, psd_x, psd_y, psd_z):
        self.freq_bins = freq_bins
        self.psd_sum = psd_sum
        self.psd_x = psd_x
        self.psd_y = psd_y
        self.psd_z = psd_z

def _most_significant_bit(N):
    res = 1
    while N > 1:
        N >>= 1
        res <<= 1
    return res

def _smooth(x, N):
    s = np.r_[x[N-1:0:-1], x, x[-2:-N-1:-1]]
    w = np.blackman(N)
    return np.convolve(w / w.sum(), s, mode='valid')

def calc_freq_response(data):
    N = data.shape[0]
    T = data[-1,0] - data[0,0]
    SAMPLING_FREQ = N / T
    # Round up to the nearest power of 2 for faster FFT
    M = _most_significant_bit(2 * int(SAMPLING_FREQ * WINDOW_T_SEC - 1))
    # 1.75 constant is just some guess for blackman window smoothing
    smooth_window = int(1.75 * SAMPLING_FREQ / MAX_FREQ)
    if N <= M or N <= smooth_window:
        return None

    # Interpolate input data to fixed sampling rate and smooth it
    t = np.linspace(data[0,0], data[-1,0], N)
    ax = _smooth(np.interp(t, data[:,0], data[:,1]), smooth_window)
    ay = _smooth(np.interp(t, data[:,0], data[:,2]), smooth_window)
    az = _smooth(np.interp(t, data[:,0], data[:,3]), smooth_window)

    mlab = matplotlib.mlab
    # Calculate PSD (power spectral density) of vibrations per window per
    # frequency bins (the same bins for X, Y, and Z)
    px, fx, _ = mlab.specgram(ax, Fs=SAMPLING_FREQ, NFFT=M, noverlap=M//2,
                              window=np.blackman(M), detrend='mean', mode='psd')
    py, fy, _ = mlab.specgram(ay, Fs=SAMPLING_FREQ, NFFT=M, noverlap=M//2,
                              window=np.blackman(M), detrend='mean', mode='psd')
    pz, fz, _ = mlab.specgram(az, Fs=SAMPLING_FREQ, NFFT=M, noverlap=M//2,
                              window=np.blackman(M), detrend='mean', mode='psd')

    # Pre-smooth PSD between consecutive windows
    px_avg = (px[:,:-2] + px[:,1:-1] + px[:,2:]) * (1./.3)
    py_avg = (py[:,:-2] + py[:,1:-1] + py[:,2:]) * (1./.3)
    pz_avg = (pz[:,:-2] + pz[:,1:-1] + pz[:,2:]) * (1./.3)
    psd = (px_avg + py_avg + pz_avg).max(axis=1)
    px_max = px_avg.max(axis=1)
    py_max = py_avg.max(axis=1)
    pz_max = pz_avg.max(axis=1)
    return CalibrationData(fx, psd, px_max, py_max, pz_max)

def plot_frequency(data, logname):
    calibration_data = calc_freq_response(np.array(data))
    freqs = calibration_data.freq_bins
    psd = calibration_data.psd_sum[freqs <= MAX_FREQ]
    px = calibration_data.psd_x[freqs <= MAX_FREQ]
    py = calibration_data.psd_y[freqs <= MAX_FREQ]
    pz = calibration_data.psd_z[freqs <= MAX_FREQ]
    freqs = freqs[freqs <= MAX_FREQ]

    fig, ax = matplotlib.pyplot.subplots()
    ax.set_title("Accelerometer data (%s)" % (logname,))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density')

    ax.plot(freqs, psd, label='X+Y+Z', alpha=0.6)
    ax.plot(freqs, px, label='X', alpha=0.6)
    ax.plot(freqs, py, label='Y', alpha=0.6)
    ax.plot(freqs, pz, label='Z', alpha=0.6)

    ax.grid(True)
    fontP = matplotlib.font_manager.FontProperties()
    fontP.set_size('x-small')
    ax.legend(loc='best', prop=fontP)
    fig.tight_layout()
    return fig


######################################################################
# Startup
######################################################################

def setup_matplotlib(output_to_file):
    global matplotlib
    if output_to_file:
        matplotlib.rcParams.update({'figure.autolayout': True})
        matplotlib.use('Agg')
    import matplotlib.pyplot, matplotlib.dates, matplotlib.font_manager
    import matplotlib.ticker

def main():
    # Parse command-line arguments
    usage = "%prog [options] <log>"
    opts = optparse.OptionParser(usage)
    opts.add_option("-o", "--output", type="string", dest="output",
                    default=None, help="filename of output graph")
    opts.add_option("-r", "--raw", action="store_true",
                    help="graph raw accelerometer data")
    options, args = opts.parse_args()
    if len(args) != 1:
        opts.error("Incorrect number of arguments")

    # Parse data
    data = parse_log(args[0])

    # Draw graph
    setup_matplotlib(options.output is not None)
    if options.raw:
        fig = plot_accel(data, args[0])
    else:
        fig = plot_frequency(data, args[0])

    # Show graph
    if options.output is None:
        matplotlib.pyplot.show()
    else:
        fig.set_size_inches(8, 6)
        fig.savefig(options.output)

if __name__ == '__main__':
    main()

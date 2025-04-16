import gradio as gr
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis
import joblib
import os
import tempfile

def bandpass_filter(signal, fs, lowcut=0.5, highcut=50, order=2):
    nyquist = 0.5 * fs
    highcut = min(highcut, nyquist - 0.1)
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def notch_filter(signal, fs, notch_freq=50, quality_factor=30):
    nyquist = 0.5 * fs
    notch = notch_freq / nyquist
    if notch >= 1.0:
        return signal
    b, a = iirnotch(notch, quality_factor)
    return filtfilt(b, a, signal)

def preprocess_signal(signal, fs):
    return notch_filter(bandpass_filter(signal, fs), fs)

def normalize_signal(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

def extract_features_from_signal(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal)
    }

def save_plot(time_axis, signal, title):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_axis, signal, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    plot_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(plot_file.name, dpi=150)
    plt.close(fig)
    return plot_file.name

def process_uploaded_files(files):
    temp_dir = tempfile.mkdtemp()
    uploaded = {}

    for file in files:
        basename = os.path.basename(file.name)
        with open(os.path.join(temp_dir, basename), 'wb') as f_out, open(file.name, 'rb') as f_in:
            f_out.write(f_in.read())
        uploaded[basename] = os.path.join(temp_dir, basename)

    ecg_path = ppg_path = None
    for name in uploaded:
        if name.endswith('.hea'):
            record_name = os.path.splitext(name)[0]
            full_path = os.path.join(temp_dir, record_name)
            try:
                record = wfdb.rdrecord(full_path)
                desc = record.sig_name[0].lower()
                if 'ecg' in desc and not ecg_path:
                    ecg_path = full_path
                elif 'ppg' in desc and not ppg_path:
                    ppg_path = full_path
            except:
                continue

    if not ecg_path or not ppg_path:
        return "❌ Could not identify ECG or PPG signal from uploaded files.", None, None, None

    record_ecg = wfdb.rdrecord(ecg_path)
    ecg_signal = preprocess_signal(record_ecg.p_signal[:, 0], record_ecg.fs)
    fs_ecg = record_ecg.fs

    record_ppg = wfdb.rdrecord(ppg_path)
    ppg_signal = preprocess_signal(record_ppg.p_signal[:, 0], record_ppg.fs)
    fs_ppg = record_ppg.fs

    if fs_ecg != fs_ppg:
        t_ppg = np.arange(len(ppg_signal)) / fs_ppg
        t_ecg = np.arange(len(ecg_signal)) / fs_ecg
        max_time = min(t_ppg[-1], t_ecg[-1])
        t_ppg = t_ppg[t_ppg <= max_time]
        t_ecg = t_ecg[t_ecg <= max_time]
        ppg_signal = interp1d(t_ppg, ppg_signal[:len(t_ppg)], fill_value="extrapolate")(t_ecg)
        ecg_signal = ecg_signal[:len(ppg_signal)]
        fs = fs_ecg
    else:
        fs = fs_ecg

    normalized_ecg = normalize_signal(ecg_signal)
    normalized_ppg = normalize_signal(ppg_signal)
    fused_signal = (normalized_ecg + normalized_ppg) / 2
    time_axis = np.arange(len(fused_signal)) / fs

    ecg_img = save_plot(time_axis, normalized_ecg, "Normalized ECG Signal")
    ppg_img = save_plot(time_axis, normalized_ppg, "Normalized PPG Signal")
    fused_img = save_plot(time_axis, fused_signal, "Fused Signal (ECG + PPG)")

    features = extract_features_from_signal(fused_signal)
    model = joblib.load("./model/my_model_optimized.pkl")
    X_new = np.array([list(features.values())])
    y_pred = model.predict(X_new)[0]
    y_proba = model.predict_proba(X_new)[0]

    emoji = "✅ Good" if y_pred == 1 else "❌ Poor"
    result_text = f"""
    <h2 style='font-size: 36px; margin-bottom: 10px;'>🧠 <b>Prediction Result</b></h2>
    <p style='font-size: 26px;'>Prediction: <b>{emoji}</b></p>
    <p style='font-size: 22px;'>Probabilities → Good: <b>{y_proba[1]:.2f}</b>, Poor: <b>{y_proba[0]:.2f}</b></p>
    """
    return result_text, ecg_img, ppg_img, fused_img

# ---------- Gradio Interface ----------
iface = gr.Interface(
    fn=process_uploaded_files,
    inputs=gr.File(file_types=[".dat", ".hea"], file_count="multiple", label="Upload ECG & PPG .dat and .hea files"),
    outputs=[
        gr.Markdown(label="Prediction Result"),
        gr.Image(label="Normalized ECG Signal"),
        gr.Image(label="Normalized PPG Signal"),
        gr.Image(label="Fused Signal")
    ],
    title="🧠 ECG + PPG Signal Fusion and Prediction",
    description="Upload your ECG and PPG signals (.dat and .hea). The system will automatically detect, process, and classify the fused signal.",
    allow_flagging="never"
)

iface.launch()

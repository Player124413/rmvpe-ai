import os
import traceback
import sys
import parselmouth
import numpy as np
import logging
import multiprocessing
from multiprocessing import set_start_method
from typing import List, Tuple
from lib.audio import load_audio
import pyworld
import torch
import torch.nn as nn

now_dir = os.getcwd()
sys.path.append(now_dir)

# Set spawn method for multiprocessing to work with CUDA
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

logging.getLogger("numba").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_extraction")

exp_dir = sys.argv[1]
log_file = f"{exp_dir}/extract_f0_feature.log"
f = open(log_file, "a+")


def printt(strr: str):
    """Print and log messages"""
    print(strr)
    f.write(f"{strr}\n")
    f.flush()
    logger.info(strr)


n_p = int(sys.argv[2])
f0method = sys.argv[3]


class RNNSmoother(nn.Module):
    """RNN-based smoother for F0 contours"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # [seq_len, 1]
        x = x.unsqueeze(0)   # [1, seq_len, 1]
        rnn_out, _ = self.rnn(x)
        output = self.fc(rnn_out.squeeze(0))
        return output.squeeze(-1)


class FeatureInput:
    def __init__(self, samplerate=16000, hop_size=160):
        self.fs = samplerate
        self.hop = hop_size
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        
        # Initialize models lazily when needed
        self.model_rmvpe = None
        self.rnn_smoother = None
        self.device = None

    def _initialize_device(self):
        """Initialize device in each process separately"""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            printt(f"Using device: {self.device}")

    def _load_rmvpe_model(self):
        """Lazy load RMVPE model"""
        if self.model_rmvpe is None:
            self._initialize_device()
            from lib.rmvpe import RMVPE
            printt("Loading RMVPE model...")
            self.model_rmvpe = RMVPE("rmvpe.pt", is_half=False, device=self.device)
    
    def _load_rnn_smoother(self):
        """Lazy load RNN smoother"""
        if self.rnn_smoother is None:
            self._initialize_device()
            printt("Initializing RNN smoother...")
            self.rnn_smoother = RNNSmoother().to(self.device)
            # Load pretrained weights if available
            smoother_path = os.path.join(now_dir, "rnn_smoother.pt")
            if os.path.exists(smoother_path):
                self.rnn_smoother.load_state_dict(torch.load(smoother_path, map_location=self.device))
            self.rnn_smoother.eval()

    def noise_reduction(self, f0: np.ndarray) -> np.ndarray:
        """Simple noise reduction for F0 contour"""
        if len(f0) == 0:
            return f0
            
        window_size = 5
        if len(f0) > window_size:
            padded = np.pad(f0, (window_size//2, window_size//2), mode='edge')
            f0_smoothed = np.zeros_like(f0)
            for i in range(len(f0)):
                window = padded[i:i+window_size]
                f0_smoothed[i] = np.median(window[window > 0]) if np.any(window > 0) else 0
            return f0_smoothed
        return f0

    def volume_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume"""
        if len(audio) == 0:
            return audio
            
        rms = np.sqrt(np.mean(audio**2))
        return audio * (0.1 / rms) if rms > 0 else audio

    def compute_f0_rmvpe_plus(self, x: np.ndarray) -> np.ndarray:
        """Enhanced RMVPE+ F0 extraction with post-processing"""
        self._load_rmvpe_model()
        
        # Volume normalization
        x = self.volume_normalization(x)
        
        # Get base F0 from RMVPE
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        
        # Skip RNN smoothing if no valid F0 values
        if np.all(f0 <= 0):
            return f0
            
        # Only load RNN smoother if needed
        if np.any(f0 > 0):
            self._load_rnn_smoother()
            # Convert to tensor for RNN processing
            f0_tensor = torch.from_numpy(f0.astype(np.float32)).to(self.device)
            
            # RNN smoothing
            with torch.no_grad():
                f0_smoothed = self.rnn_smoother(f0_tensor).cpu().numpy()
        else:
            f0_smoothed = f0
        
        # Noise reduction
        f0_processed = self.noise_reduction(f0_smoothed)
        
        # Remove potential NaN values
        return np.nan_to_num(f0_processed, nan=0.0)

    def compute_f0(self, path: str, f0_method: str) -> np.ndarray:
        """Compute F0 using specified method"""
        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        
        if f0_method == "pm":
            time_step = 160 / 16000 * 1000
            f0 = (
                parselmouth.Sound(x, self.fs)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method in ["harvest", "dio"]:
            method = pyworld.harvest if f0_method == "harvest" else pyworld.dio
            f0, t = method(
                x.astype(np.double),
                fs=self.fs,
                f0_ceil=self.f0_max,
                f0_floor=self.f0_min,
                frame_period=1000 * self.hop / self.fs,
            )
            f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.fs)
        elif f0_method == "rmvpe":
            self._load_rmvpe_model()
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        elif f0_method == "rmvpe+":
            f0 = self.compute_f0_rmvpe_plus(x)
        else:
            raise ValueError(f"Unknown f0 method: {f0_method}")
        
        if f0 is None:
            raise RuntimeError(f"F0 extraction failed for {path}")
        
        return f0[:p_len]  # Ensure correct length

    def coarse_f0(self, f0: np.ndarray) -> np.ndarray:
        """Convert F0 to coarse representation"""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1

        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f"F0 coarse values out of range: min={f0_coarse.min()}, max={f0_coarse.max()}"
        )
        return f0_coarse

    def process_file(self, inp_path: str, opt_path1: str, opt_path2: str, f0_method: str):
        """Process a single audio file"""
        try:
            if (
                os.path.exists(opt_path1 + ".npy")
                and os.path.exists(opt_path2 + ".npy")
            ):
                return
            
            printt(f"Processing {inp_path} with {f0_method}")
            
            # Compute F0 features
            featur_pit = self.compute_f0(inp_path, f0_method)
            
            # Save features
            np.save(opt_path2, featur_pit, allow_pickle=False)  # nsf
            coarse_pit = self.coarse_f0(featur_pit)
            np.save(opt_path1, coarse_pit, allow_pickle=False)  # ori
            
        except Exception as e:
            printt(f"Failed to process {inp_path}: {str(e)}\n{traceback.format_exc()}")

    def go(self, paths: List[Tuple[str, str, str]], f0_method: str):
        """Process multiple files"""
        if not paths:
            printt("No files to process")
            return
        
        printt(f"Processing {len(paths)} files with {f0_method}")
        
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            self.process_file(inp_path, opt_path1, opt_path2, f0_method)
            
            # Progress reporting
            if (idx + 1) % max(len(paths) // 5, 1) == 0 or (idx + 1) == len(paths):
                printt(f"Progress: {idx + 1}/{len(paths)}")


def main():
    printt(f"Starting F0 extraction with arguments: {sys.argv}")
    
    try:
        # Initialize feature input in main process
        featureInput = FeatureInput()
        
        # Prepare paths
        inp_root = f"{exp_dir}/1_16k_wavs"
        opt_root1 = f"{exp_dir}/2a_f0"
        opt_root2 = f"{exp_dir}/2b-f0nsf"

        os.makedirs(opt_root1, exist_ok=True)
        os.makedirs(opt_root2, exist_ok=True)
        
        # Collect files to process
        paths = []
        for name in sorted(os.listdir(inp_root)):
            if "spec" in name:
                continue
            inp_path = f"{inp_root}/{name}"
            opt_path1 = f"{opt_root1}/{name}"
            opt_path2 = f"{opt_root2}/{name}"
            paths.append((inp_path, opt_path1, opt_path2))
        
        # Process in parallel
        ps = []
        chunk_size = len(paths) // n_p
        for i in range(n_p):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < n_p - 1 else len(paths)
            chunk = paths[start:end]
            
            if not chunk:
                continue
                
            p = multiprocessing.Process(
                target=featureInput.go,
                args=(chunk, f0method),
            )
            ps.append(p)
            p.start()
        
        for p in ps:
            p.join()
            
        printt("F0 extraction completed successfully")
        
    except Exception as e:
        printt(f"Fatal error in main: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)
        
    finally:
        f.close()


if __name__ == "__main__":
    main()

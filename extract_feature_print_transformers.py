import os, sys, traceback

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# device=sys.argv[1]
n_part = int(sys.argv[2])
i_part = int(sys.argv[3])
if len(sys.argv) == 6:
    exp_dir = sys.argv[4]
    version = sys.argv[5]
else:
    i_gpu = sys.argv[4]
    exp_dir = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    version = sys.argv[6]
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import transformers
from transformers import HubertModel

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

f = open("%s/extract_f0_feature.log" % exp_dir, "a+")


def printt(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


printt(sys.argv)
model_path = "Hubert"  # Local directory containing model files
config_path = os.path.join(model_path, "config.json")  # Path to config.json
model_file_path = os.path.join(model_path, "pytorch_model.bin")  # Path to pytorch_model.bin

printt(exp_dir)
wavPath = "%s/1_16k_wavs" % exp_dir
outPath = (
    "%s/3_feature256" % exp_dir if version == "v1" else "%s/3_feature768" % exp_dir
)
os.makedirs(outPath, exist_ok=True)


# wave must be 16k, hop_size=320
def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


# HuBERT model
class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)
model = HubertModelWithFinalProj.from_pretrained(model_path)
model = model.to(device)
if device not in ["mps", "cpu"]:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavPath)))[i_part::n_part]
n = max(1, len(todo) // 10)  # 最多打印十条
if len(todo) == 0:
    printt("no-feature-todo")
else:
    printt("all-feature-%s" % len(todo))
    for idx, file in enumerate(todo):
        try:
            if file.endswith(".wav"):
                wav_path = "%s/%s" % (wavPath, file)
                out_path = "%s/%s" % (outPath, file.replace("wav", "npy"))

                if os.path.exists(out_path):
                    continue

                feats = readwave(wav_path, normalize=False)
                padding_mask = torch.BoolTensor(feats.shape).fill_(False)
                inputs = {
                    "input_values": (
                        feats.half().to(device)
                        if device not in ["mps", "cpu"]
                        else feats.to(device)
                    ),
                    "attention_mask": padding_mask.to(device),
                    "output_hidden_states": True
                with torch.no_grad():
                    outputs = model(**inputs)
                    if version == "v1":
                        # Для v1 берём 9-й слой и применяем final_proj
                        hidden_states = outputs.hidden_states[9]
                        feats = model.final_proj(hidden_states)
                    else:
                        # Для других версий берём 12-й слой
                        hidden_states = outputs.hidden_states[12]
                        feats = hidden_states

                feats = feats.squeeze(0).float().cpu().numpy()
                if np.isnan(feats).sum() == 0:
                    np.save(out_path, feats, allow_pickle=False)
                else:
                    printt("%s-contains nan" % file)
                if idx % n == 0:
                    printt("now-%s,all-%s,%s,%s" % (len(todo), idx, file, feats.shape))
        except:
            printt(traceback.format_exc())
    printt("all-feature-done")

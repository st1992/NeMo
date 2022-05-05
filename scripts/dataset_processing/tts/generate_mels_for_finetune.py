# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path
import os
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from nemo.collections.tts.torch.helpers import BetaBinomialInterpolator
from nemo.collections.tts.models import FastPitchModel


def get_args():
    """
    mel-generator checkpoint
    """
    parser = argparse.ArgumentParser(
        description="Generate mel spectrograms with pretrained FastPitch model, and feed into HiFi-GAN model for training."
    )
    parser.add_argument("--fastpitch-model", required=True, type=Path)
    parser.add_argument("--json-file", required=True, type=Path)

    args = parser.parse_args()
    return args


def load_wav(audio_file):
    with sf.SoundFile(audio_file, 'r') as f:
        samples = f.read(dtype='float32')
    return samples.transpose()


def main():
    args = get_args()
    ckpt_path = args.fastpitch_model
    manifest_path = args.json_file

    # load pretrained FastPitch model checkpoint
    spec_model = FastPitchModel.load_from_checkpoint(ckpt_path)
    spec_model.eval()
    device = spec_model.device

    beta_binomial_interpolator = BetaBinomialInterpolator()

    # Get records from the training manifest
    # manifest_path = "./6097_manifest_train_dur_5_mins_local.json"
    records = []
    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))

    # save_dir = Path("./6097_manifest_train_dur_5_mins_local_mels")
    save_dir = os.path.splitext(manifest_path)[0]
    os.makedirs(save_dir, exist_ok=True)
    # save_dir.mkdir(exist_ok=True, parents=True)

    # Generate a spectrograms (we need to use ground truth alignment for correct matching between audio and mels)
    for i, r in enumerate(records):
        audio = load_wav(r["audio_filepath"])
        audio = torch.from_numpy(audio).unsqueeze(0).to(device)
        audio_len = torch.tensor(audio.shape[1], dtype=torch.long, device=device).unsqueeze(0)

        # Again, our finetuned FastPitch model doesn't use multiple speakers,
        # but we keep the code to support it here for reference
        if spec_model.fastpitch.speaker_emb is not None and "speaker" in r:
            speaker = torch.tensor([r['speaker']]).to(device)
        else:
            speaker = None

        with torch.no_grad():
            if "normalized_text" in r:
                text = spec_model.parse(r["normalized_text"], normalize=False)
            else:
                text = spec_model.parse(r['text'])

            text_len = torch.tensor(text.shape[-1], dtype=torch.long, device=device).unsqueeze(0)
            spect, spect_len = spec_model.preprocessor(input_signal=audio, length=audio_len)

            # Generate attention prior and spectrogram inputs for HiFi-GAN
            attn_prior = torch.from_numpy(
                beta_binomial_interpolator(spect_len.item(), text_len.item())
            ).unsqueeze(0).to(text.device)

            spectrogram = spec_model.forward(
                text=text,
                input_lens=text_len,
                spec=spect,
                mel_lens=spect_len,
                attn_prior=attn_prior,
                speaker=speaker,
            )[0]

            save_path = os.path.join(save_dir, f"mel_{i}.npy")
            np.save(save_path, spectrogram[0].to('cpu').numpy())
            r["mel_filepath"] = str(save_path)

    hifigan_manifest_path = "hifigan_train.json"
    with open(hifigan_manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

            
if __name__ == "__main__":
    main()
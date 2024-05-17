import os
from pathlib import Path
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import torch

# set dirs
homedir = str(Path.home())
workdir = os.path.join(homedir, 'code/repo/parler-tts')
if os.getcwd() != workdir:
    os.chdir(workdir)
print('current dir: {}'.format(os.getcwd()))

# set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device: {}'.format(device))

# change the default huggingface home dir (for saving space in home dir)
os.environ['HF_HOME'] = os.path.join(homedir, 'data/.cache/huggingface')

# model_path = "parler-tts/parler_tts_mini_v0.1" # huggingface model path
model_path = os.path.join(os.getcwd(), 'models', 'parler_tts_mini_v0.1') # local model path
model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "Hey, how are you doing today?"
description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, " + \
              "in a very confined sounding environment with clear audio quality. She speaks very slow."
output_filepath = os.path.join(os.getcwd(), 'output', 'parler_tts_out.wav')

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write(output_filepath, audio_arr, model.config.sampling_rate)
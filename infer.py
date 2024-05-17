# infer the online demo samples
# 
# reference: https://www.text-description-to-speech.com/
# samples to be inferred: samples in the 2nd part (controling specific attributes)
#
# Zhenhao Ge, 2024-05-08

import os
from pathlib import Path
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import pyloudnorm as pyln
import torch
import argparse
import json
import time
import numpy as np
from numpy.linalg import norm
import math
import re
import subprocess
from librosa.util import normalize
from pyannote.audio import Inference

# set dirs
homedir = str(Path.home())
workdir = os.path.join(homedir, 'code/repo/parler-tts')
if os.getcwd() != workdir:
    os.chdir(workdir)
print('current dir: {}'.format(os.getcwd()))

spkr_embedding = Inference("pyannote/embedding", window="whole")

def convert_nan(entry):
    for k in entry.keys():
        cond1 = isinstance(entry[k], str)
        if not cond1:
            cond2 = math.isnan(entry[k])
        else:
            cond2 = False
        # print('key: {}, cond1: {}, cond2: {}'.format(k, cond1, cond2))
        if (not cond1) and cond2:
            entry[k] = ''
    return entry

def convert_float2str(file_group):
    for k in file_group.keys():
        cond1 = isinstance(file_group[k], np.floating)
        cond2 = isinstance(file_group[k], float)
        if cond1 or cond2:
            # print('{}: {} is a np.float or float'.format(k, file_group[k]))
            file_group[k] = '{:.2f}'.format(file_group[k])
        elif isinstance(file_group[k], list):
            # print('{}: {} is a list'.format(k, file_group[k]))
            for i in range(len(file_group[k])):
                cond1 = isinstance(file_group[k][i], np.floating)
                cond2 = isinstance(file_group[k][i], float)
                if cond1 or cond2:
                    file_group[k][i] = '{:.2f}'.format(file_group[k][i])
    return file_group                  

def get_audio(speaker_path, meter, sample_rate=16000):

    audio, sr = sf.read(speaker_path) # load audio (shape: samples, channels)
    # assert sr == sample_rate, 'sampling rate is {} (should be {})'.format(sr, sample_rate)
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    loudness = meter.integrated_loudness(audio) # measure loudness
    audio = pyln.normalize.loudness(audio, loudness, -20.0)
    return audio

def cos_sim(A,B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def get_hostname():
    hostname = subprocess.check_output('hostname').decode('ascii').rstrip()
    return hostname

def get_gpu_info(device):
    line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
    lines = line_as_bytes.decode("ascii").split('\n')
    lines = [line for line in lines if line != '']
    line = lines[device]
    string = re.sub("\(.*?\)","()", line).replace('()','').strip()
    return string

def extract_spkr_embedding(wav, sample_rate):

     # normalize wav
    wav = normalize(wav) * 0.95
    # convert wav (np.ndarray:(nsamples,) -> torch.Tensor: (1,nsamples))
    wav = torch.FloatTensor(wav).unsqueeze(0)
    # get speaker embedding (np.ndarry: (spkr_emb_dim:512,))
    speaker_embedding = spkr_embedding({'waveform': wav, 'sample_rate': sample_rate})

    return speaker_embedding

def generate(model, tokenizer, description, prompt, device):

    # durs = [0 for _ in range(3)]
    time_start = time.time()

    input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # time_cp0 = time.time()
    # durs[0] = time_cp0 - time_start

    generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

    # time_cp1 = time.time()
    # durs[1] = time_cp1 - time_cp0

    audio_arr = generation.cpu().numpy().squeeze()

    time_end = time.time()
    # durs[2] = time_end - time_cp1
    dur = time_end - time_start

    # print('dur1: {:.3f}, dur2: {:.3f}, dur3: {:.3f}'.format(*durs))
    print('time elapsed: {:.3f}'.format(dur))

    return audio_arr, dur

def parse_args():
    usage = 'usage: infer the online demo samples'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--manifest-file', type=str, help='manifest json file')
    parser.add_argument('--model-path', type=str, help='parler-tts model path')
    parser.add_argument('--output-path', type=str, help='output path')
    parser.add_argument('--num-copies', type=int, default=-1, 
        help='number of syn output audio copies for the same setting')
    parser.add_argument('--device', type=int, help='gpu device id, -1 to use cpu')
    return parser.parse_args()
    
if __name__ == '__main__':

    # runtime mode
    args = parse_args()

    # # interactive mode
    # args = argparse.ArgumentParser()

    # work_path = os.getcwd() # e.g., '/home/users/zge/code/repo/parler-tts'
    # data_path = os.path.join(work_path, 'examples', 'parler-tts-demo')
    # args.manifest_file = os.path.join(data_path, 'manifest.json')
    # args.model_path = os.path.join(work_path, 'models', 'parler_tts_mini_v0.1')
    # args.num_copies = 2
    # args.output_path = os.path.join(data_path, 'wav')
    # args.device = 0

    # sanity check
    assert os.path.isfile(args.manifest_file), \
        'manifest file: {} does not exist!'.format(args.manifest_file)
    assert os.path.isdir(args.model_path), \
        'model path: {} does not exist!'.format(args.model_path)

    # create the output path (if not exist)
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        print('created output path: {}'.format(args.output_path))
    else:
        print('output path {} already exist'.format(args.output_path))

    # get hostname and GPU info
    hostname = get_hostname()
    gpu_info = get_gpu_info(args.device)      

    # specify the GPU/CPU device
    if args.device == -1:
        device = 'cpu'
        print('using CPU to infer')
    else:    
        if torch.cuda.is_available():
            device = 'cuda:{}'.format(args.device)
        else:
            raise Exception('CUDA is not available, switch device to CPU (-1) instead!')

    # print out input arguments
    print('manifest file: {}'.format(args.manifest_file))
    print('model path: {}'.format(args.model_path))
    print('number of copies: {}'.format(args.num_copies))
    print('output path: {}'.format(args.output_path))
    print('device: {}'.format(device))

    # load model and tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # get meter for sound normalization
    meter = pyln.Meter(model.config.sampling_rate) # create BS.1770 meter (default block size: 400ms)

    # set max new tokens to control the output length
    model.generation_config.max_new_tokens = 2000

    # read manifest file
    lines = open(args.manifest_file).readlines()

    # get # of samples and # of output files (# of samples X # of copies)
    num_samples = len(lines)
    num_output_files = num_samples * args.num_copies
    print('# of samples: {}'.format(num_samples))
    print('# of output files: {}'.format(num_output_files))

    file_groups = [{} for _ in range(num_samples)]
    for i in range(num_samples):

        # get current entry
        entry = json.loads(lines[i])

        # convert nan to '' in entry values
        entry = convert_nan(entry)

        # get prompt and description
        prompt = entry['prompt']
        description = entry['description']

        rtfs, durs_syn, durs_proc, syn_wavs = [], [], [], []
        for j in range(args.num_copies):

            # get output file path
            output_filename = '{}_{:02d}.wav'.format(entry['file-id'], j+1)
            output_filepath = os.path.join(args.output_path, output_filename)

            print('generating syn wav: {} ...'.format(output_filepath))

            # generate syn audio (also measure the processing time)
            audio_arr, dur_proc = generate(model, tokenizer, description, prompt, device)
            sf.write(output_filepath, audio_arr, model.config.sampling_rate)

            # get RTF
            dur_syn = len(audio_arr)/model.config.sampling_rate
            rtf = dur_proc / dur_syn

            # append RTFs, durations (syn, proc), and syn wavs
            rtfs.append(rtf)
            durs_syn.append(dur_syn)
            durs_proc.append(dur_proc)
            syn_wavs.append(output_filepath)

        # compute speaker similarity between file[j] and file[j+1], then take the average
        sss = []
        for j in range(args.num_copies-1):
            basename1 = os.path.basename(syn_wavs[j])
            basename2 = os.path.basename(syn_wavs[j+1])
            print('computing spkr similarity between {} and {}'.format(basename1, basename2))
            wav1 = get_audio(syn_wavs[j], meter, sample_rate=model.config.sampling_rate)
            wav2 = get_audio(syn_wavs[j+1], meter, sample_rate=model.config.sampling_rate)
            syn_spkr_embedding1 = extract_spkr_embedding(wav1, model.config.sampling_rate)
            syn_spkr_embedding2 = extract_spkr_embedding(wav2, model.config.sampling_rate)
            sss.append(cos_sim(syn_spkr_embedding1, syn_spkr_embedding2))
        sss_avg = np.mean(sss)

        # collect all info into file_groups
        file_groups[i] = entry
        file_groups[i].update({'sss': sss_avg, 'rtfs': rtfs, 'durs-syn': durs_syn,
                               'durs-proc': durs_proc, 'syn_wavs': syn_wavs,
                               'hostname': hostname, 'gpu': gpu_info})

    # for i in range(num_samples):
    #     file_groups[i].update({'hostname': hostname, 'gpu': gpu_info})

    # print the avg. speaker similarity score
    sss_mean = np.mean([float(file_group['sss']) for file_group in file_groups])
    print('mean spkr similarity score: {:.3f}'.format(sss_mean))

    # print the avg. rtf
    rtf_mean = np.mean([np.mean([float(rtf) for rtf in file_group['rtfs']]) for file_group in file_groups])
    print('mean RTF: {:.3f}'.format(rtf_mean))

    # convert float to string in file_groups
    for i in range(num_samples):
        file_groups[i] = convert_float2str(file_groups[i])

    # write file group to a json file for future reference
    data_path = os.path.abspath(os.path.join(args.output_path, os.pardir))
    file_group_jsonfile = os.path.join(data_path, 'file_groups.json')
    with open(file_group_jsonfile, 'w') as fp:
        json.dump(file_groups, fp, indent=2)
    print('wrote file group json file to {}'.format(file_group_jsonfile))                        

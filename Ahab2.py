import streamlit as st
import pandas as pd
from scipy.io import wavfile
import os
import time
import argparse
from tqdm import tqdm
import pandas as pd
import ketos.neural_networks.dev_utils.detection as det
from ketos.audio.audio_loader import AudioFrameLoader
from ketos.neural_networks.resnet import ResNetInterface
from ketos.neural_networks.dev_utils.detection import process, save_detections, merge_overlapping_detections
from ketos.audio.spectrogram import MagSpectrogram
import matplotlib.pyplot as plt
from PIL import Image

header = st.container()
inference = st.container()
spectrogram = st.container()

with header:
    st.title("Welcome to Captain Ahab: the whale detector")
    image = Image.open('whale.jpg')

    st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    input_file = st.file_uploader('Please upload your .wav file', type='wav')
    input = str(input_file.name)


with inference:
    st.header("Inference:")
    st.text("Please choose the hyperparameters")   

    treshold_val = st.slider("Detection Treshold", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    step_size = st.slider("Step Size", min_value=0.0, max_value=3.0, value=1.5, step=0.5)
    buffer_bool = st.radio("Buffer?", ["yes", "no"])
    batches = st.selectbox("Batch Size", options=[32,64,128])
    merge = st.radio("Merge overlapping detections", [True, False])
    window = st.selectbox("Length of score averaging window", options=[1,3,5])
    
with spectrogram:
    st.header('Magnitude Spectrogram')
    spec = MagSpectrogram.from_wav(path=input, window=0.2, step=0.01)
    fig = spec.plot() #create the figure
    fig.savefig('spectro.png')

    spectro_display = Image.open('spectro.png')
    caption = 'spectrogram of: ' + input
    st.image(spectro_display, caption=caption)


# load the classifier and the spectrogram parameters
model, audio_repr = ResNetInterface.load_model_file('narw.kt', './narw_tmp_folder', load_audio_repr=True)
spec_config = audio_repr[0]['spectrogram']


# initialize the audio loader
audio_loader = AudioFrameLoader(frame=spec_config['duration'], step=step_size, filename=input, repres=spec_config)

# process the audio data
detections = process(provider=audio_loader, model=model, batch_size=batches, buffer=buffer_bool, threshold=treshold_val, group='store_true', win_len=window, progress_bar='store_false')
if merge == True:
    detections = merge_overlapping_detections(detections)

save_detections(detections=detections, save_to='detections.csv')












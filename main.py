import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from pyannote.audio import Pipeline
import torch
import pandas as pd
from pydub import AudioSegment
from pathlib import Path
import random
import shutil
import base64
import requests

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

emoji_list = [
    ":sparkles:",    # ✨
    ":rocket:",      # 🚀
    ":fire:",        # 🔥
    ":tada:",        # 🎉
    ":sunglasses:",  # 😎
    ":robot:",       # 🤖
    ":hourglass:",   # ⏳
    ":bulb:",        # 💡
    ":dart:",        # 🎯
    ":trophy:"       # 🏆
]

def markdown():
    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def huggingface_query(filename, url):
    with open(filename, "rb") as f:
        data = f.read()
    headers = {
    	"Accept" : "application/json",
    	"Authorization": "Bearer {}".format(HUGGINGFACE_TOKEN),
    	"Content-Type": "audio/flac" 
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()

st.title("Testing Transcribing Feature")
markdown()

st.write("Status message:")
status_msg = st.empty()
status_msg.write("Step 1 of 4: Waiting for audio :speaking_head_in_silhouette:")

markdown()
c1 = st.container()
with c1:
    upload_option = st.radio("Select an option to upload audio", ["Record Audio", "Upload Audio"], horizontal=True)

c2 = st.container()
with c2:
    if upload_option == "Record Audio":
        audio_bytes = audio_recorder(pause_threshold=10.0, text="Click Mic Start/Stop Record", icon_size="4x")
    else:
        audio_bytes = st.file_uploader("Upload an audio file", type=["wav"])


markdown()
text_area = st.empty()
text_area.text_area("Transcribed Text", "", height=400)

if audio_bytes:
    
    status_msg.write("Step 2 of 4: processing audio :speaker:")

    with c2: 
        st.audio(audio_bytes, format="audio/wav")
        if hasattr(audio_bytes, 'name'):
            suffix = Path(audio_bytes.name).suffix
            audio_bytes = audio_bytes.read()
        else:
            suffix=".wav"

    audio_filepath = "audio{}".format(suffix)
    diarization_filepath = "diarization.csv"
    audio_chunk_filepath = "audio_chunks"
    text_filepath = "text.txt"
    if os.path.exists(audio_chunk_filepath):
        shutil.rmtree(audio_chunk_filepath)
    os.makedirs(audio_chunk_filepath)

    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)

    if os.path.exists(text_filepath):
        os.remove(text_filepath)

    with open(audio_filepath, mode='wb') as f:
        f.write(audio_bytes)

    st.markdown(get_binary_file_downloader_html(audio_filepath, "Audio File"), unsafe_allow_html=True)

    if suffix==".mp3":
        audio_segment = AudioSegment.from_mp3(audio_filepath)
    else:
        audio_segment = AudioSegment.from_wav(audio_filepath)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGINGFACE_TOKEN)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline.to(torch.device(device))
    # model = whisper.load_model("turbo")

    status_msg.write("Step 3 of 4: Diarizing Audio :card_index_dividers:")
    diarization = pipeline(audio_filepath)
    diarization_list = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarization_list.append({"start": turn.start, "end": turn.end, "speaker": int(speaker[-2:])})
    diarization_df = pd.DataFrame(diarization_list)

    if diarization_df.shape[0]>0:
        diarization_df.iloc[0]["start"] = 0
        diarization_df['group'] = (diarization_df["speaker"] != diarization_df["speaker"].shift()).cumsum()
        agg_functions = {'start': 'min', 'end': 'max', 'speaker': 'first'}
        diarization_grouped = diarization_df.groupby(diarization_df['group']).aggregate(agg_functions)
        conversations = diarization_grouped.shape[0]
        
        transciption = ''
        for index in diarization_grouped.index:
            status_msg.write("Step 4 of 4: Transcribing conversation {0} of {1} {2}".format(index, conversations, emoji_list[random.randint(0, 9)]))
            t1 = diarization_grouped.loc[index]["start"] * 1000 #Works in milliseconds
            t2 = diarization_grouped.loc[index]["end"] * 1000
            speaker = diarization_grouped.loc[index]["speaker"]
            newAudio = audio_segment[t1:t2]
            newAudio.export(audio_chunk_filepath+'/{}.wav'.format(index), format="wav") 
            output = huggingface_query(audio_chunk_filepath+'/{}.wav'.format(index), 
                                    "https://anh0r8n4iu12yqni.us-east-1.aws.endpoints.huggingface.cloud")
            if "text" in output:
                text = output["text"]
            else:
                text = output["error"]+ " Error Occured. Retry!!"
                status_msg.write("Error Occured. Retry!!")

            diarization_grouped.loc[index, "text"] = text
            transciption += "Speaker {0}: {1}\n".format(int(speaker+1), text)
            text_area.text_area("Transcribed Text", transciption, height=400)

        diarization_grouped.to_csv(diarization_filepath)    
        status_msg.write("Transcribing Completed!!")

        with open(text_filepath, mode='w') as f:
            f.write(transciption)

        st.markdown(get_binary_file_downloader_html(text_filepath, "Text File"), unsafe_allow_html=True)

    else:
        status_msg.write("No speech found in Audio, Try again with new Audio")
    
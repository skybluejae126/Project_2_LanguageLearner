# 📖 Project_2_TTSLanguageLearning 📖

## ✎ About ✎
  - Python, TTS, RVC, VOICEVOX, DTW, cuda

## ✎ Purpose of this project ✎
  - One of the biggest challenges in learning a foreign language is accurately pronouncing words with the correct intonation and accent. While native speakers’ pronunciations can serve as a reference, it is often difficult to mimic their tone precisely.

  - This project allows users to generate speech using their own TTS model and practice speaking by mimicking the generated audio. The similarity between the TTS-generated speech and the user's recorded voice is evaluated with a score, and differences can be visualized using waveform graphs.

## ✎ Goals ✎
🚨 **Currently, this project supports only Japanese language learning.** 🚨

  - Various TTS models exist, but most are trained primarily on specific languages. To effectively learn a language, a suitable TTS model trained on that language is required.

- **Current Features:**
  - Supports **Japanese** language learning.
  - Uses **VOICEVOX** to generate Japanese TTS audio.
- **Future Plans:**
  - Expand support for **English, Korean, Chinese**, and other languages.
  - Develop a **web-based version** for ease of use.

## ✎ File Structure ✎
```
/Learning_Language_Myvoice
│── /RVC_output       # Generated RVC audio
│── /User_voice       # User's recorded voice
│── /VOX_output       # Generated VOICEVOX audio
│── /models           # User's TTS models
│── .gitignore
│── Voice_similarity.py  # Compares RVC audio and user's voice, assigns score
│── main.py              # Generates RVC audio from input text (Japanese)
│── requirements.txt
│── LICENSE
```

## ✎ How It Works ✎
### **0. Place the User's TTS Model**
- Place the user's TTS model in `/models`.
- The default model name is `mitest.pth`.
- You can create a your own TTS model easily by using [RVC-beta](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md).

### **1. Generate TTS Audio (main.py)**
1. Input a Japanese sentence.
2. Generate a TTS voice using **VOICEVOX**.
   - `speaker=1` for female voice, `speaker=11` for male voice.
   - Adjust `voicespeed` for optimal speed.
3. Convert the **VOICEVOX TTS** output into the **user's TTS model** using **RVC**.

### **2. Evaluate Speech Similarity (Voice_similarity.py)**
1. Listen to the **RVC-generated** voice and try to mimic it.
2. Record your voice and save it as `myvoice.wav` in `/User_voice`.
3. Run the program to compare the voices using the **DTW (Dynamic Time Warping) algorithm**.
4. The program assigns a similarity score (out of 100) and displays a waveform comparison.

## ✎ How to Use ✎
### **Preprocessing**
#### **1. Install CUDA**
- This project was developed in a **CUDA 11.8** environment.
- It is recommended to install **CUDA 11.8** for best compatibility.

#### **2. Set Up VOICEVOX Docker**
This project **does not modify VOICEVOX** but uses its Docker image.
- Refer to: [VOICEVOX Engine](https://github.com/VOICEVOX/voicevox_engine?tab=readme-ov-file)
- **GPU version:**
```sh
docker pull voicevox/voicevox_engine:nvidia-latest
docker run --rm --gpus all -p '127.0.0.1:50021:50021' voicevox/voicevox_engine:nvidia-latest
```

#### **3. Set Up RVC**
This project uses the **RVC Python package**.
- Refer to: [RVC Python](https://github.com/daswer123/rvc-python)
```sh
py -3.10 -m venv venv
venv\Scripts\activate
pip install rvc-python
pip install torch==2.1.1+cu118 torchaudio==2.1.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### **4. Install Dependencies**
- Required for **speech similarity scoring** and **waveform visualization**.
```sh
pip install -r requirements.txt
```

### **Usage**
#### **1. Place the Model**
- Create a **TTS model** using RVC and place it in `/models`.
- The model should be named **`mitest.pth`**.

#### **2. Generate TTS Audio**
```sh
python main.py
```
- Input a **Japanese** sentence in `main.py` and execute the script.
- Generates **VOICEVOX** and **RVC-adjusted** TTS audio.

#### **3. Compare Speech Similarity**
1. Listen to the generated **RVC TTS** and try to mimic it.
2. Record your voice and save it as `myvoice.wav` in `/User_voice`.
3. Run the comparison script:
```sh
python Voice_similarity.py
```
4. The program assigns a **similarity score** and displays waveform graphs.

## ✎ Dependencies ✎
- Python 3.10.x
- cuda 11.8

## ✎ Reference Projects ✎
- **VOICEVOX Engine** ([GitHub](https://github.com/VOICEVOX/voicevox_engine/tree/master?tab=readme-ov-file))
  - **License:** LGPL v3 (No modifications made)
- **RVC** ([GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md))
- **RVC Python** ([GitHub](https://github.com/daswer123/rvc-python))

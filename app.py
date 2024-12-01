from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model

def generate_music_tensors(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )
    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )
    return output[0]

def save_audio(samples: torch.Tensor):
    """Saves audio samples to a .wav file."""
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def plot_waveform(samples, sample_rate):
    """Plots the waveform of the audio."""
    plt.figure(figsize=(10, 4))
    plt.plot(samples.squeeze().numpy())
    plt.title("Waveform")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    st.pyplot(plt)

def plot_spectrogram(samples, sample_rate):
    """Plots the spectrogram of the audio."""
    samples = samples.squeeze().numpy()
    spectrogram = np.abs(np.fft.rfft(samples, axis=-1))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)

    plt.figure(figsize=(10, 4))
    plt.semilogy(frequencies, spectrogram)
    plt.title("Spectrogram")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    st.pyplot(plt)

st.set_page_config(
    page_icon="🎵",
    page_title="Music Gen with Analysis"
)

def main():
    # Sidebar for inputs and the generate button
    with st.sidebar:
        st.header("Music Generator Inputs")
        st.write("Configure your input to generate music 🎼")
        text_area = st.text_area("Enter your description:")
        time_slider = st.slider("Select time duration (seconds):", 1, 20, 10)
        generate_button = st.button("Generate Music")

    # Main content on the right
    st.title("🎶 Text-to-Music Generator with Analysis")
    st.write("""
    This application uses **Meta's Audiocraft library** and the **MusicGen-Small model** to generate music based on your input. 
    Add a description of the music you want, select the duration, and click the **Generate Music** button to experience AI-generated melodies.
    """)

    if generate_button and text_area and time_slider:
        st.subheader("Generated Music and Analysis")
        st.json({
            'Your Description': text_area,
            'Selected Time Duration (in Seconds)': time_slider
        })

        # Generate music
        music_tensors = generate_music_tensors(text_area, time_slider)
        save_audio(music_tensors)

        # Save and play generated music
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Download Generated Audio'), unsafe_allow_html=True)

        # Music analysis
        st.write("### Music Analysis")
        sample_rate = 32000  # MusicGen default sample rate
        samples = music_tensors.squeeze(0)

        # Show basic properties
        st.write("**Audio Tensor Shape:**", samples.shape)
        st.write("**Sample Rate:**", sample_rate)
        st.write("**Duration (seconds):**", samples.shape[-1] / sample_rate)
        st.write("**Mean Amplitude:**", torch.mean(samples).item())
        st.write("**Amplitude Range:**", torch.min(samples).item(), "to", torch.max(samples).item())
        st.write("**Standard Deviation of Amplitude:**", torch.std(samples).item())

        # Visualizations
        st.write("#### Waveform")
        plot_waveform(samples, sample_rate)

        st.write("#### Spectrogram")
        plot_spectrogram(samples, sample_rate)

if __name__ == "__main__":
    main()

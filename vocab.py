import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.title("WebRTC Minimal Mic Test")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.count = 0

    def recv_audio_frame(self, frame):
        self.count += 1
        return frame

webrtc_ctx = webrtc_streamer(
    key="mic-test",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx:
    st.write("State:", webrtc_ctx.state)
    if webrtc_ctx.audio_processor:
        st.write("Frames processed:", webrtc_ctx.audio_processor.count)
    else:
        st.write("Audio processor not started yet.")

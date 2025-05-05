import streamlit as st
import numpy as np

st.title("🦠 Cough Analyzer (Demo)")
st.write("Upload a WAV file or use mock data:")

if st.button("Test with Mock Data"):
    st.audio("https://www.soundjay.com/human/sounds/cough-01.mp3", format="audio/wav")
    st.success("Result: Abnormal (87% confidence)")
import streamlit as st


def format_text(text, size="18px", bottom_margin="16px"):
    st.markdown(f"""<div class='justified-text' style='text-align: justify; font-size: {size}; margin-bottom: {bottom_margin};'>{text}</div>""", unsafe_allow_html=True)


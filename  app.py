import tempfile
import streamlit as st
import cv2
import time
import torch
import numpy as np
from model import vision_model
from PIL import Image
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)


def stream_output(model, web_address = None, video = None):
    start_button_pressed = st.button("Detect")

    if web_address is not None:
        cap = cv2.VideoCapture(web_address)
    elif video is not None:
        cap = cv2.VideoCapture(video)
    else:
        cap = cv2.VideoCapture(0)

    frame_placeholder = st.empty()

    if start_button_pressed:
        stop_button_pressed = st.button("Stop")

    while cap.isOpened() and start_button_pressed and not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break

        annotated_frame = get_annoteted_img(frame, model)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(annotated_frame,channels="RGB")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()


def get_annoteted_img(img, model):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    start = time.time()
    annotated_frame = model.predict(img)[0].plot()
    end = time.time()
    print(img.shape)
    log_speed(device, img.shape[:2], end-start)
    return cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)


def main():
    check_speed_file()

    model = vision_model(
        signs_model_path= '/Users/sasamarjanovic/Projects/traffic_vision/models/signs_models/m_model/best.pt',
        lights_model_path= '/Users/sasamarjanovic/Projects/traffic_vision/models/lights_models/m_model/best.pt',
        device = device
    )

    st.set_page_config(page_title="Traffic Vision")
    st.title("Traffic Vision - Object Detection")
    st.write("Choose input type and provide input for object detection.")

    input_type = st.selectbox("Select Input Type", ["Image", "Images", "Video", "Webcam", "IP Camera"])

    uploader = st.empty()

    if input_type == "Image":
        uploaded_image = uploader.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            annotated_img = get_annoteted_img(Image.open(uploaded_image), model)
            st.image(annotated_img, caption=uploaded_image.name)

    elif input_type == "Images":
        uploaded_images = uploader.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_images is not None:
            columns = st.columns(3)
            for i, image in enumerate(uploaded_images):
                annotated_img = get_annoteted_img(Image.open(image), model)
                with columns[i % 3]:
                    st.image(annotated_img, caption=image.name)

    elif input_type == "Video":
        uploaded_video = uploader.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            stream_output(model, video=tfile.name)

    elif input_type == "Webcam":
        stream_output(model)

    elif input_type == "IP Camera":
        ip_address = st.text_input("Enter IP Camera Address")
        if st.button("Connect") and ip_address is not None:
            stream_output(model, web_address=ip_address)


    # Instructions for user
    st.markdown("Note: For webcam or IP camera, make sure to close the stream using the 'Stop' button before changing the input type.")


if __name__ == "__main__":
    main()
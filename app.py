import streamlit as st 
import matplotlib.image as mpimg
import os 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import base64
import pickle
import json
from streamlit_lottie import st_lottie
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="DDoS Detector in Network", page_icon=":warning:",layout="wide")
loaded_model = pickle.load(open("trained_ddos.sav",'rb'))
chs = ' '

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_coding = load_lottiefile("Anim.json")


def video_bg(url: str):
    video_file = open(url, 'rb')
    video_bytes = video_file.read()
    base64_video = base64.b64encode(video_bytes).decode('utf-8')

    video_html = f'''
        <style>
            #myVideo {{
                position: fixed;
                right: 0;
                bottom: 0;
                min-width: 100%;
                min-height: 100%;
            }}
            .content {{
               position: fixed;
               bottom: 0;
               background: rgba(0, 0, 0, 0.5); /* Black background with opacity */
               color: #f1f1f1; /* White text color */
               width: 100%;
               padding: 20px;
            }}
        </style>
        <video autoplay muted loop id="myVideo">
            <source src="data:video/mp4;base64,{base64_video}" type="video/mp4">
            Your browser does not support HTML5 video.
        </video>
    '''
    return video_html

# Embed the video as a background
video_url = 'vids.mp4'  # Replace with the path to your video file
st.markdown(video_bg(video_url), unsafe_allow_html=True)

# Add content on top of the video
st.markdown("<h1 style='text-align: center; color: white; background-color: black; opacity: 80%'>Welcome to DDoS Attack Detector </h1><br><br>", unsafe_allow_html=True)

with st.container():
    left_column,right_column = st.columns((2,1))
    with right_column:
        st_lottie(
        lottie_coding,
        speed = 1,
        reverse = False,
        loop = True,
        quality="high",height=300, width=None, key=None)
    with left_column:
        def predict(input_data):
            base_dir = 'C:\\Users\\91982\\Desktop\\DDoS_Model'
            train_dir = os.path.join(base_dir, 'Train')
            validation_dir = os.path.join(base_dir, 'Validation')
            # Directory with our training cat pictures
            train_traffic_dir = os.path.join(train_dir, 'Traffic')
            # Directory with our training dog pictures
            train_normal_dir = os.path.join(train_dir, 'Normal')
            # Directory with our validation cat pictures
            validation_traffic_dir = os.path.join(validation_dir, 'Traffic')
            # Directory with our validation dog pictures
            validation_normal_dir = os.path.join(validation_dir, 'Normal')
            nrows = 4
            ncols = 4
            fig = plt.gcf()
            fig.set_size_inches(ncols*4, nrows*4)
            pic_index = 100
            train_traffic_fnames = os.listdir(  )
            train_normal_fnames = os.listdir( train_normal_dir )
            next_traffic_pix = [os.path.join(train_traffic_dir, fname) 
                for fname in train_traffic_fnames[ pic_index-8:pic_index] 
               ]
            next_normal_pix = [os.path.join(train_normal_dir, fname) 
                for fname in train_normal_fnames[ pic_index-8:pic_index]
               ]
            for i, img_path in enumerate(next_traffic_pix+next_normal_pix):
                # Set up subplot; subplot indices start at 1
                sp = plt.subplot(nrows, ncols, i + 1)
                sp.axis('Off') # Don't show axes (or gridlines)
                img = mpimg.imread(img_path)
            train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
            # Note that the validation data should not be augmented!
            test_datagen = ImageDataGenerator( rescale = 1.0/255. )
            train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary', target_size = (224, 224))
            # Flow validation images in batches of 20 using test_datagen generator
            validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'binary', target_size = (224, 224))
            vgghist = loaded_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 3)
            validation_generator.reset()
            y_pred =loaded_model.predict(validation_generator, steps=validation_generator.n // validation_generator.batch_size + 1, verbose=1)
            y_pred_binary = (y_pred > 0.5).astype(int)
            y_true = validation_generator.classes
            # confusion matrix
            conf_mat = confusion_matrix(y_true, y_pred_binary)
            class_names=['Normal','Traffic']
            predicted_class=class_names[int(np.round(y_pred[input_data]))]
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Predicted: {predicted_class}")
            plt.show()
        def main():
            Ins = st.text_input("Enter the RequestID")
            if st.button("Check"):
                chs = predict(Ins)
            st.success(chs)
        if __name__=='__main__':
            main()


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
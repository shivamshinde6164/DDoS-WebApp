import os
import base64
import zipfile
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.image as mpimg
import streamlit as st
import numpy as np

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
# Load the VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16
# Path to the directory containing training and validation data
base_dir = 'C:\\Users\\91982\\Desktop\\DDoS_Model'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
train_traffic_dir = os.path.join(train_dir, 'Traffic')

# Directory with our training dog pictures
train_normal_dir = os.path.join(train_dir, 'Normal')

# Directory with our validation cat pictures
validation_traffic_dir = os.path.join(validation_dir, 'Traffic')

# Directory with our validation dog pictures
validation_normal_dir = os.path.join(validation_dir, 'Normal')

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')
for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])
# Add your data generators
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(224, 224))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(224, 224))

vgghist = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 1)
validation_generator.reset()
y_pred = model.predict(validation_generator, steps=validation_generator.n // validation_generator.batch_size + 1, verbose=1)
y_pred_binary = (y_pred > 0.5).astype(int)
y_true = validation_generator.classes

# confusion matrix
conf_mat = confusion_matrix(y_true, y_pred_binary)

# Function to load and display images
def load_and_display_images():
    nrows = 4
    ncols = 4
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index = 100
    train_traffic_dir = os.path.join(train_dir, 'Traffic')
    train_normal_dir = os.path.join(train_dir, 'Normal')
    train_traffic_fnames = os.listdir(train_traffic_dir)
    train_normal_fnames = os.listdir(train_normal_dir)
    next_traffic_pix = [os.path.join(train_traffic_dir, fname) for fname in train_traffic_fnames[pic_index - 8:pic_index]]
    next_normal_pix = [os.path.join(train_normal_dir, fname) for fname in train_normal_fnames[pic_index - 8:pic_index]]
    for i, img_path in enumerate(next_traffic_pix + next_normal_pix):
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')
        img = mpimg.imread(img_path)
        plt.imshow(img)
    return fig

# Function to display the classification report
def display_classification_report():
    st.subheader("Classification Report:")
    st.text(classification_report(y_true, y_pred_binary))

# Function to display training and validation accuracy
def display_accuracy():
    train_accuracy = vgghist.history['acc']
    validation_accuracy = vgghist.history['val_acc']
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
    plt.plot(range(1, len(validation_accuracy) + 1), validation_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

# Function to display confusion matrix
def display_confusion_matrix():
    st.subheader("Confusion Matrix:")
    return plt.gcf()

# Function to predict and display the output
def predict_and_display_output():
    class_names = ['Normal', 'Traffic']
    predicted_class = class_names[int(np.round(y_pred[11]))]
    st.subheader(f"Predicted: {predicted_class}")
    return plt.gcf()

# Function to evaluate the model

def evaluate_model(validation_generator):
    eval_result = model.evaluate(validation_generator)
    accuracy = eval_result[1]
    st.write(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Get evaluation results
    loss = eval_result[0]
    accuracy = eval_result[1]

    st.write(f"Loss: {loss:.4f}")
    st.write(f"Accuracy: {accuracy:.4f}")


# Load the model and evaluate
st.markdown("<h1 style='text-align: center; color: white; background-color: black; opacity: 80%'>Welcome to DDoS Attack Detector </h1><br><br>", unsafe_allow_html=True)
fig = load_and_display_images()
st.pyplot(fig)
evaluate_model(validation_generator)
predict_and_display_output()


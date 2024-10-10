import streamlit as st
import pickle
import tensorflow as tf

# Load the pickled model
pickle_m = pickle.load(open('regmodel1.pkl', 'rb'))

# Streamlit app code
def main():
    st.title('Image Prediction App')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Read and process the image
        image = tf.image.decode_image(uploaded_file.read(), channels=3)  # Decode as RGB
        image = tf.image.resize(image, (256, 256))  # Resize to the model's input shape
        image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale
        image = image / 255.0  # Normalize the image
        test_img = tf.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the trained model
        prediction = pickle_m.predict(test_img)

        # Display prediction result
        st.write('Prediction:', prediction[0][0])

        # Interpret the prediction
        if prediction[0][0] > 0.5:
            st.write("Prediction: Pneumonia")
        else:
            st.write("Prediction: Normal")

if __name__ == '__main__':
    main()

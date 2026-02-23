Files Description

**`first.ipynb`**: A Jupyter Notebook used for training the model. It handles data loading, preprocessing, model definition, and the training loop. This is just to show how we trained the data and evaluated the trained model using CONFUSION MATRIX.
**`realtime_emotion_detection.py`**: The inference script. It loads the trained model, captures video from the webcam, detects faces using Haar Cascades, and overlays the predicted emotion and confidence score.

**`emotion_cnn.pth`**: (Generated after training) The file containing the saved weights of the trained model.
**`app_streamlit.py`** This is the python script written just to make the model visible using streamlit
## Prerequisites

To run this project, you will need Python installed along with the following libraries:
1. Torch
2. opencv
3. numpy
4. matplotlib
5. Torchvision

also can be installed by the following command,

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## Dataset
fer2013

### Step 1:
Open `first.ipynb` and if you run this, it will run the preprocessing cells and train the model and also at the end will show the precision and accuracy of the model.for this you need the fer2013 in the same folder.
# you can skip this step too

### Step 2:

open the `realtime_emotion_detection.py` file and run it (just have to make sure that the saved model weight file "emotion_cnn.pth" have to be in the same folder as this python file)
Then the file will run and the video will open automatically.

### Note you have to press q to exit the cam or terminate the execution

Also you can run the model but as streamlit for which,

### Step 3:

After downloading the folder locate  `app_streamlit.py` through your terminal:
Process:
1. copy the folder path
2. open terminal
3. type (cd 'paste the path')
4. type (streamlit run app_streamlit.py)
then the cam will open
(Streamlit also can be used to send the link of the frontend to the other person so that they can access the model without having to download the code and run it manually)


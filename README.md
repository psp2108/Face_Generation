# Computerized Suspect Face Generation System

### Environment:
- Tensorflow: 2.3.0
- Tensorflow-gpu: 2.3.0
- Python 3.7.8
- Numpy: 1.18.5
- Matplotlib: 3.3.0
- Pillow: 7.2.0
- tqdm
- Cuda: 10.1 
- CudNN: 7.6.x
- Jupyter: 1.0.0
- Notebook: 6.1.1
- Flask: 1.1.2
- opencv-contrib-python: 4.3.0.38
- dlib: 19.22.0
- cmake: 3.18.4.post1
- face-recognition: 1.3.0 opencv


### References:
- [Version compatibility](https://www.tensorflow.org/install/source_windows)
- [Steps to install Cuda and Cudnn](https://towardsdatascience.com/nstalling-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)
- [Dataset ](https://www.kaggle.com/jessicali9530/celeba-dataset?select=img_align_celeba)

---

### How to Train
1. Data preprocessing
    - Image Resizing to 128x128
    - Normalizing Attributes
    - Analysing the CSVs
    - Deleting Images which reflected improper attributes
    - Merge Attributes
    - Attribute Selection
    - Deleting the Records where the face is not clearly visible
    - Balancing Male to Female Ratio
2. Train network
3. Test Run Model

### How to Execute
1. Go to src folder
2. Run app.py to initiate server
3. Open templates/UI/home.html in browser

[Demo Video](https://www.youtube.com/watch?v=PK9s0FKDLIA)
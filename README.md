# Computerized Suspect Face Generation System

### Environment:
- Tensorflow: 2.3.0
- Tensorflow-gpu: 2.3.0
- Python 3.7.8
- Numpy: 1.18.5
- Matplotlib: 3.3.0
- Cuda: 10.1 
- CudNN: 7.6.x
- Jupyter: 1.0.0
- Notebook: 6.1.1

### References:
- [Version compatibility](https://www.tensorflow.org/install/source_windows)
- [Steps to install Cuda and Cudnn](https://towardsdatascience.com/nstalling-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)
- [Dataset ](https://www.kaggle.com/jessicali9530/celeba-dataset?select=img_align_celeba)

---

### How to Execute
1. Data preprocessing
    - Crop images to 128x128
    - Merge CSV
    - Remove Blurred
    - Manual Filtering

2. Data normalization
3. Train network
4. Run Model
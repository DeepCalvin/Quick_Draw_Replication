# QuickDraw CNN Game 🎨🤖

A Quick! Draw! replication using CNN that classifies 12 images from your drawings!

---

## Features

- **Data Preprocessing**
  • Loads 12 classes from this link: https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?inv=1&invt=Ab43ow&prefix=&forceOnObjectsSortingFiltering=false&pli=1
  • Normalizes pixel values and splits into train/test sets
- **Model Training**
  • Custom CNN built with Keras (TensorFlow backend)
  • 4 convolutional blocks + fully connected at the end
  • Batch normalization and dropout for robust learning 
  • Configurable training (batch size 64, 12 epochs)  
- **Real‐Time Drawing Interface**
  • Pygame window where you sketch with your mouse
  • Press **Space** to clear, **Esc** to quit
  • Live “Guess: … (confidence)” display
- **Save and Load**
  • After training, save your model as `quickdraw_cnn_model.h5`
  • Interface loads the saved model for instant play

---

## Classes (12)

1. airplane  
2. basketball  
3. cat  
4. eye  
5. axe  
6. stairs  
7. t-shirt  
8. bicycle  
9. car  
10. stop_sign  
11. tree  
12. mug  



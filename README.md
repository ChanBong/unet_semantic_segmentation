# Semantic Segmentation using UNET

We use the U-Net architechture to do semantic segmentation on **Helen** Dataset

## Architecture


![alt text](./assets/unet_architecture.png)

------
  
## Dataset

**Helen** dataset consists 2000 train images, masks and 100 test images, masks with 11 classes.

The dataset has been resized to 256,256 for both images and segmentation masks.

  
![alt text](./assets/image_7.png) ![alt text](./assets/label_7.PNG)



![alt text](./assets/image_45.png)![alt text](./assets/label_45.PNG)

----


## Model predictions on test set

![alt text](./assets/test_predictions_1.png)
![alt_text](./assets/test_predictions_2.png)



## Uses

- To retrain 
```bash 
python3 train.py 
```

- To get class wise f1 scores
```bash
python3   path/to/f1_score/    path/to/test/labels    path/to/test/preds  path/to/labels_names.txt
```
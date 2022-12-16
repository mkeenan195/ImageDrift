# Mitigating Covariate Shift with Style Transfer

## Resources

- Style transfer code based on the [Neural Transfer Using Pytorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html) tutorial by Alexis Jacq.
- YOLO object detection model based on [yolov5](https://github.com/ultralytics/yolov5) implemented by ultralytics.
- [Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset) published by [Nikolov et al.](https://openreview.net/forum?id=LjjqegBNtPi).

## Project Description


## Repository Description
The code can be divided into two stages: style transfer and object detection. 

The file [stylized_datasets.py](stylized_datasets.py) creates the baseline and style-augmented datasets used for training object detection models. The code for implementing the neural style transfer algorithm is in the [style](style) directory. Examples of the style-augmented datasets are in [datasets](datasets) and [datasets_highstyle](datasets_highstyle).


## Instructions

### Style Transfer
To create the baseline and style-augmented training datasets and the test datasets, first create a new directory that contains a subdirectory with the annotated images from the [Long-term Thermal Drift Dataset](https://www.kaggle.com/datasets/ivannikolov/longterm-thermal-drift-dataset). Then run the following Python command using the path to the new directory.
```
python style_transfer.py --datadir <PATH_TO_DATASET_DIRECTORY> --content_weight 10
```
The `content_weight` option controls the degree of style added to content images by changing the weight of the content loss. For the low-style experiments we set `content_weight` equal to 10 and for the high-style experiments we set it equal to 1.

### Object Detection


## Results

## References

- Jacq, Alexis. “Neural Transfer Using Pytorch.” https://pytorch.org/tutorials/advanced/neural_style_tutorial.html.

- Glenn Jocher, et al., Ultralytics, yolov5: v6.2 - YOLOv5 Classification Models, Reproducibility, ClearML and Deci.ai integrations (v6.2), (2022)
- Nikolov, Ivan Adriyanov, et al. "Seasons in Drift: A Long-Term Thermal Imaging Dataset for Studying Concept Drift." Thirty-fifth Conference on Neural Information Processing Systems, (2021).
- Zheng, Xu, et al. "Stada: Style transfer as data augmentation." arXiv preprint arXiv:1909.01056, (2019).

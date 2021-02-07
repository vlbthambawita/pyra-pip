# pyra-pytorch

This is a dataset class suporting Pytorch datasets. This implementation is based on the augmentaiton method discuss in the following paper "Pyramid-Focus-Augmentation: Medical Image Segmentation
with Step-Wise Focus" and the original github repository. 

[PAPER](https://arxiv.org/pdf/2012.07430v1.pdf) | [Original implementaiton in GitHub](https://vlbthambawita.github.io/PYRA/)



```latex
@article{thambawita2020pyramid,
  title={Pyramid-Focus-Augmentation: Medical Image Segmentation with Step-Wise Focus},
  author={Thambawita, Vajira and Hicks, Steven and Halvorsen, P{\aa}l and Riegler, Michael A},
  journal={arXiv preprint arXiv:2012.07430},
  year={2020}
}
```

# How to use:

### Install the package,
```bash
pip install pyra-pytorch
```

### Create a dataset with gird sizes which are going to be used as augmentation in the training process. If you want to get only the original mask, then, you have to pass image size as the gird size. 

```python
from pyra_pytorch import PYRADataset

dataset = PYRADataset("./image_path", # image folder
                      "./masks_path", # mask folder - files´s names of this folder should have image names as prefix to find correct image and mask pairs.
                      img_size = 256,  # height and width of image to resize
                      grid_sizes=[2,4,8,16,32,64,128,256] , # Gird sizes to use as grid augmentation. Note that, the image size after resizing ()
                      transforms = None
                      )
'''
./image_path" --> image folder

./masks_path" --> mask folder - files´s names of this folder should have image names as prefix to find correct image and mask pairs.

img_size = 256 --> height and width of image to resize

grid_sizes=[2,4,8,16,32,64,128,256]  --> Gird sizes to use as grid augmentation. Note that, the image size after resizing (in this case, it is 256) shoud be divisible by these grid sizes.
                      
transforms = None --> Other type of transformations using in Pytorch. 

'''
```

# Sample ipython notebook

[notebook](https://github.com/vlbthambawita/pyra-pytorch/blob/main/tutorial/load_data_with_PYRA.ipynb)


# Contact us:
```
vajira@simula.no | michael@simula.no
```
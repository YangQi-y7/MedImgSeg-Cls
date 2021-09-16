# Medical Image Segmentation & Classification
**A basic demo for Medical Image Processing with [LiTS](https://arxiv.org/abs/1901.04056) (Liver Tumor Segmentation) dataset.**
## Quick Deployment
1. Download LiTS dataset from [zenodo](https://zenodo.org/record/3734294) or [academictorrents](https://academictorrents.com/details/27772adef6f563a1ecc0ae19a528b956e6c803ce).
2. copy this repository.
3.  go to `parameters.py` to change data paths.
4. run the preprocessing files from `dataset_prepare`.
5. Now you good to go.

**Note that** in step 4,  after `1_cut_patch.py`, you still need to manually split the dataset and copy them into `train` and `test` folder.

**Also** only use a fraction of the dataset at first otherwise it can take a really long time to prepare the data.

## Explore different tasks using different models
+ For segmentation, both 3D and 2D images can be used, using a basic U-net setup.
+ For classification a pre-tarined ResNet18 is used. **note that the labels are randomly generated**, so you can't get valid results from this dataset. Only see this as a test run.

Image sizes and Models used are as follows:
![models](img/models.png)

## basic deep learning pipline
![markdown](https://www.mdeditor.cn/images/logos/markdown.png "markdown")

## Metrics
![markdown](https://www.mdeditor.cn/images/logos/markdown.png "markdown")
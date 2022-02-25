# Tiny COCO dataset

Generate a tiny coco dataset for training debug. 
The training and test sets each contain 50 images and the corresponding instance, keypoint, and capture tags.
The dataset file structure as follows:

```python
tiny_coco
    |-annotations
    |   |-instances_train2017.json
    |   |-instances_val2017.json
    |   |-...
    |
    |-train2017
    |   |-000000005802.jpg
    |   |-000000005803.jpg
    |   |-...
    |
    |-val2017
    |   |-000000005802.jpg
    |   |-000000005803.jpg
    |   |-...
```


## split script
Reference from [zhihu](https://zhuanlan.zhihu.com/p/423898204). 
You can re-slice the data size through the original coco dataset(18G) or the current tiny coco dataset
# 3D Point Cloud Generation With ManiDiff
<img src="manidiff/fig1.png" alt="Image text" width="300" height="200"/>
Implementation of Shape Generation and Completion Through The ”Many Manifold” Hypothesis for 3D Shapes.

## Requirements:
Make sure the following environments are installed.
    
    python==3.9
    pytorch==1.10.2
    torchvision==0.11.3
    cudatoolkit==11.8.0
    matplotlib==2.2.3
    tqdm==4.66.1
    open3d==0.17.0
    scipy==1.10.1

## Training

### Data:
ShapeNet can be downloaded here.
Put the downloaded data as ./data/ShapeNetCore.v2.PC15k or edit the pointflow entry in ./datasets/data_path.py for the ShapeNet dataset path.

### Pretrained models:
Pretrained models can be downloaded here.

### Demo:
run python demo.py, will load the model generate a car point cloud. 

### Evaluation:
download the test data from here, unzip and put it as ./datasets/test_data/
run python ./script/compute_score.py 

## Results:
Some generation and completion results are as follows.
<img src="manidiff/result.jpg" alt="Image text" >

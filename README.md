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


### train diffusion 
We take the encoder and decoder trained on the data as usual (without conditioning input), and when training the diffusion prior, we feed the clip image embedding as conditioning input: the shape-latent prior model will take the clip embedding through AdaGN layer.
require the vae checkpoint trained above
require the rendered ShapeNet data, you can render yourself or download it from here
put the rendered data as ./data/shapenet_render/ or edit the clip_forge_image entry in ./datasets/data_path.py
the img data will be read under ./datasets/pointflow_datasets.py with the render_img_path, you may need to cutomize this variable depending of the folder structure
run bash ./script/train_prior_clip.sh $NGPU

### Data:
ShapeNet can be downloaded here.
Put the downloaded data as ./data/ShapeNetCore.v2.PC15k or edit the pointflow entry in ./datasets/data_path.py for the ShapeNet dataset path.

### Pretrained models:
Pretrained models can be downloaded [here]().

### Demo:
run python demo.py, will load the model generate a car point cloud. 

### Evaluation:
download the test data from here, unzip and put it as ./datasets/test_data/
run python ./script/compute_score.py 

## Results:
Some generation and completion results are as follows.

<img src="manidiff/result.png" alt="Image text" width="600" height="800">

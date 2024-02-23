# IAPC for SFDA Object Detection

# Code

## Prerequisites
- We use Python 3.6, PyTorch 1.9.0 (CUDA 10.2 build).
- The codebase is built on [Detectron](https://github.com/facebookresearch/detectron2).

```angular2

conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch

cd OD
pip install -r requirements.txt

## Make sure you have GCC and G++ version <=8.0
cd ..
python -m pip install -e OD

```



## Datasets
* **CitysScape, FoggyCityscape**: Download website [Cityscape](https://www.cityscapes-dataset.com/), see dataset preparation code in [DA-Faster RCNN](https://github.com/tiancity-NJU/da-faster-rcnn-PyTorch)


## Training

```angular2
python tools/train_st_sfda_net.py --config-file configs/IAPC_foggy.yaml --model-dir ./source_model/source.pth
```

## Testing

```angular2
python tools/plain_test_net.py --eval-only  --config-file configs/IAPC_foggy.yaml --model-dir $PATH TO CHECKPOINT
```

# Results

<div align="left">
<table>
  <tr>
      <td></td> 
      <th>Model</th> 
      <th>mAP</th>
  </tr>
  <tr>
      <td rowspan="2">Cityscapes-to-CityscapesFoggy</td>    
      <td ><a href="https://drive.google.com/file/d/1LUJllsbukL3DHEfZujACE6UdzPP0xqhO/view?usp=sharing" target="_blank">Source Only</a></td>  
      <td >26.3</td>
   </tr>
   <tr>
      <td ><a href="https://drive.google.com/file/d/1aZ1H9juFu5MkGl1SVe-fE45-hsTk5jtk/view?usp=sharing" target="_blank">IAPC</a></td> 
      <td >37.6</td>
   </tr>
</table>
   </div>



## Acknowledgement
This codebase is heavily borrowed from [IRG-SFDA](https://github.com/Vibashan/irg-sfda).

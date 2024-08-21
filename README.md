# GAInS: Gradient Anomaly-aware Biomedical Instance Segmentation [IEEE BIBM 2024]

## Introduction

### Abstract
Instance segmentation plays a vital role in the morphological quantification of biomedical entities such as tissues and cells, enabling precise identification and delineation of different structures. Current methods often address the challenges of touching, overlapping or crossing instances through individual modeling, while neglecting the intrinsic interrelation between these conditions.
In this work, we propose a **G**radient **A**nomaly-aware Biomedical **In**stance **S**egmentation approach **GAInS**, which leverages instance gradient information to perceive local gradient anomaly regions, thus modeling the spatial relationship between instances and refining local region segmentation. Specifically, Gradient Anomaly is defined as a kind of local directional anomaly upon multiple gradient directions or conflict of gradient directions among pixels. GAInS is firstly built on a Gradient Anomaly Mapping Module (GAMM), which encodes the radial fields of instances through window sliding to obtain instance gradient anomaly maps. To efficiently refine boundaries and regions with gradient anomaly attention, we propose an Adaptive Local Refinement Module (ALRM) with a gradient anomaly-aware loss function.
Extensive comparisons and ablation experiments in three biomedical scenarios demonstrate that our proposed GAInS outperforms other state-of-the-art (SOTA) instance segmentation methods.
### Overview
![Overview of the proposed GAInS.](overview_fig.png)
### Qualitative Result
![Qualitative Result of our GAInS and other SOTA methods.](visualization.png)
### Quantitative Result
<table>
  <thead>
    <tr>
      <th rowspan="2">Methods</th>
      <th colspan="2">ISBI2014</th>
      <th colspan="2">UCOC</th>
      <th colspan="2">Kaggle2018</th>
    </tr>
    <tr>
      <th>mAP↑</th>
      <th>AJI↑</th>
      <th>mAP↑</th>
      <th>AJI↑</th>
      <th>mAP↑</th>
      <th>AJI↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Mask R-CNN (R50)</td>
      <td>59.59</td>
      <td>76.21</td>
      <td>72.33</td>
      <td>81.29</td>
      <td>38.75</td>
      <td>54.79</td>
    </tr>
    <tr>
      <td>Mask R-CNN (R101)</td>
      <td>62.97</td>
      <td>75.15</td>
      <td>73.71</td>
      <td>80.87</td>
      <td>37.13</td>
      <td>52.80</td>
    </tr>
    <tr>
      <td>Mask Scoring R-CNN</td>
      <td>60.03</td>
      <td>71.78</td>
      <td>70.31</td>
      <td>83.21</td>
      <td>37.32</td>
      <td>52.81</td>
    </tr>
    <tr>
      <td>PISA</td>
      <td>60.84</td>
      <td>74.56</td>
      <td>73.24</td>
      <td>81.73</td>
      <td>38.09</td>
      <td>51.73</td>
    </tr>
    <tr>
      <td>Cascade R-CNN</td>
      <td>63.40</td>
      <td>52.21</td>
      <td>73.66</td>
      <td>81.09</td>
      <td>40.33</td>
      <td>54.09</td>
    </tr>
    <tr>
      <td>CondIns</td>
      <td>49.46</td>
      <td>59.79</td>
      <td>50.57</td>
      <td>66.88</td>
      <td>38.41</td>
      <td>46.50</td>
    </tr>
    <tr>
      <td>HTC</td>
      <td>62.57</td>
      <td>35.95</td>
      <td>70.32</td>
      <td>83.69</td>
      <td>37.73</td>
      <td>53.48</td>
    </tr>
    <tr>
      <td>Pointrend</td>
      <td>62.07</td>
      <td>69.45</td>
      <td>71.14</td>
      <td>31.48</td>
      <td>37.95</td>
      <td>52.63</td>
    </tr>
    <tr>
      <td>Occlusion R-CNN</td>
      <td>62.35</td>
      <td>78.64</td>
      <td>67.30</td>
      <td>83.52</td>
      <td>35.85</td>
      <td>51.81</td>
    </tr>
    <tr>
      <td>DoNet</td>
      <td>63.43</td>
      <td><strong>79.88</strong></td>
      <td>70.97</td>
      <td>82.87</td>
      <td>37.83</td>
      <td>53.96</td>
    </tr>
    <tr>
      <td>FastInst</td>
      <td>61.28</td>
      <td>71.66</td>
      <td>72.74</td>
      <td>80.12</td>
      <td>37.02</td>
      <td>50.93</td>
    </tr>
    <tr>
      <td><strong>GAInS</strong></td>
      <td><strong>63.71</strong></td>
      <td>76.82</td>
      <td>73.94</td>
      <td>83.59</td>
      <td><strong>40.63</strong></td>
      <td>53.62</td>
    </tr>
    <tr>
      <td><strong>GAInS (R101)</strong></td>
      <td>61.39</td>
      <td>77.66</td>
      <td><strong>74.48</strong></td>
      <td><strong>85.87</strong></td>
      <td>39.79</td>
      <td><strong>55.30</strong></td>
    </tr>
  </tbody>
</table>

## Dataset
For convenience, the three datasets are collected in the Google Drive folders listed below. One can downloaded directly from Google Drive or from the official websites. 
- [ISBI2014](https://drive.google.com/drive/folders/1a3_Gc4synvTUrP593L6C05II3_YK2CeP?usp=sharing)
- [UOUC](https://drive.google.com/drive/folders/192ippUdETwGp9Wrowt3oCGU_wAALN8_E?usp=sharing)
- ['cluster_nuclei' subset of Kaggle2018](https://drive.google.com/drive/folders/1o_VoeV7Ip_jLbRCeRkaMgPzTx_XnceOO?usp=sharing)

**Note:**  
1. For ISBI2014, we follow the offical division of training, validation and testing. For UOUC and Kaggle2018, the training, validation, and testing sets are divided by a ratio of 6:2:2 with random seeds. 
2. We use the 'cluster_nuclei' subset of Kaggle2018 in the paper. If needed, please download the whole Kaggle2018 dataset from its offical website.

## Installation
For detectron2 installation instructions, please refer to the [Detectron2 Installation Guide](detectron2/INSTALL.md). After installing Detectron2, the data preprocessing needs further libraries such as scipy, matpoltlib, shapely. We recommand using conda or pip install.
``` bash
pip install scipy
pip install matpoltlib
pip install shapely
``` 

## QuickStart
### Data Preprocessing
Data preprocessing is for Gradient Anomaly Map generation. Datasets like ISBI2014 where one image contains small amount of instances and faces overlapping issues, are recommonded using ```isbi_process.py```. For nuclei datasets where one image contains huge amount of instances and faces touching issues, we recommend using```nuclei_process.py```. For chromosomes which face crossing issues, we recommend using ```chrom_process.py```. The reasons they are treated differently are all about running time and a few special cases (as mentioned in the paper). 

To process, set the path of your own dataset at line 29-30 of ```isbi_process.py```, ```nuclei_process.py``` or ```chrom_process.py```. The json file requires COCO json format.
To set up hyperparameters, set ```window_size```, ```GA_factor``` and ```overlap_HL``` to proper values at line 32-34. Empirically, ```window_size``` could be set at a value of 1/20 to 1/10 of the size of an instance in the images. GA_factor is the attention rate you want to put on CTO regions, empirically in a proper interval [0.5, 2]. For ```overlap_HL```, the highlight of overlapping regions, empirically a small number is enoug h, such as 0.1 or 0.5. 

Here we provid our parameters for reference.
-  ISBI2014: window_size = 5, GA_facor = 0.5, overlap_HL = 0.1
-  UOUC: window_size = 5, GA_factor = 0.5, overlap_HL = 0.3
-  Kaggle2018: window_size = 8, GA_factor = 0.8, overlap_HL = 0.1

Additionally, set the ```target_id``` at line 40. It is the instance ID in your dataset json file. For example, the id of the cells in ISBI2014 is 0.

After setting up all the parameters, one can run the scripts. For example, 
``` bash
python isbi_process.py
python nuclei_process.py
python chrom_process.py
```
The data processing only requires CPU devices. The Gradient Anomaly Maps will be saved in the dataset folder. We also provided visualization functions in these three script. Visualize the Gradient Anomaly Maps if needed.

### Train on Your Own Datasets
This should be done after the preprocessing step. 

To train on your own data, firstly register your datasets by adding the path of your datasets at ```detectron2/detectron2/data/datasets/builtin.py``` line 57. Secondly, fill in the meta information of your datasets at ```detectron2/detectron2/data/datasets/builtin_meta.py``` line 30. 

Config file is at ```detectron2/tools/my_config.yaml```. Things you may change: 
- Line 14, 16: Name of your datasets.
- Line 220: Number of classes.
- Line 224: Path to your Gradient Anomaly Maps.
- Line 295: Pretrained model weight.
- Line 296: Output directory.
- Line 301: Learning rate.
- Line 304: Checkpoint period.
- Line 311: Batch size.
- Line 313: Maximum number of iteration.
- Line 344: Test stage evaluation period.

The training step requires GPUs. Use ```detectron2/tools/train_net.py``` for training and testing.

Training is done by
``` bash
python train_net.py --config-file my_config.yaml
```

Test is done by
``` bash
python train_net.py --config-file my_config.yaml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

Use ```detectron2/demo/demo.py``` to visualize results.
``` bash
python demo.py --config-file /path/to/config-file --input /path/to/image --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

### Our Models
We provide our models, i.e. GAInS with R50 and R101 on three datasets, totally 6 models on [One Drive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/rliuar_connect_ust_hk/EmUTQhTKlBlOt4fUy1Cs0fsBld607SLyXkf57JaZ5jBH7w?e=YDwcBA). One can download them as pre-trained models, or evaluate on the pre-trained models if needed.

**Note**
The evaluation code rely on a modified pycoco package that provides a new function iouIntUni to compute intersection over union between masks, return iou, intersection, union together. For installation of the modified pycoco package, please refer to https://github.com/Amandaynzhou/MMT-PSM.

## License
Detectron2 is released under the [Apache 2.0 license](https://github.com/DeepDoNet/DoNet/blob/master/LICENSE).
## Citation


## Acknowledgement
The code of GAInS is built on [detectron2](https://github.com/facebookresearch/detectron2) and [ORCNN](https://github.com/waiyulam/ORCNN), thanks for the Third Party Libs.

## Question
Feel free to email us if you have questions: 

Runsheng Liu (rliuar@connect.ust.hk), Hao Jiang (hjiangaz@cse.ust.hk)

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

### Dataset
For convenience, the three datasets are collected in the Google Drive folders listed below. One can downloaded directly from Google Drive or from the official websites. 
- [ISBI2014](https://drive.google.com/drive/folders/1a3_Gc4synvTUrP593L6C05II3_YK2CeP?usp=sharing)
- [UOUC](https://drive.google.com/drive/folders/192ippUdETwGp9Wrowt3oCGU_wAALN8_E?usp=sharing)
- ['cluster_nuclei' subset of Kaggle2018](https://drive.google.com/drive/folders/1o_VoeV7Ip_jLbRCeRkaMgPzTx_XnceOO?usp=sharing)

**Note:**  
1. For ISBI2014, we follow the offical division of training, validation and testing. For UOUC and Kaggle2018, the training, validation, and testing sets are divided by a ratio of 6:2:2 with random seeds. 
2. We use the 'cluster_nuclei' subset of Kaggle2018 in the paper. If needed, please download the whole Kaggle2018 dataset from its offical website.

## Installation

## QuickStart

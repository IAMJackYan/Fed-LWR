## Install

```
git clone https://github.com/IAMJackYan/Fed-LWR.git

pip install -r requirements.txt
```

## Prepare data

1) Download the retinal fundus segmentation [dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view).
   
2) Use the script to preprocess the data
   
`python preprocess_rif.py`  # Please change the script to input your data path.

## Run
`python fedlwr.py --data_root  your_data_path`

## Citation

```
@InProceedings{Yan2024fedlwr,
author="Yan, Yunlu and Zhu, Lei and Li, Yuexiang and Xu, Xinxing and Goh, Rick Siow Mong and Liu, Yong and Khan, Salman and Feng, Chun-Mei",
title="A New Perspective to Boost Performance Fairness For Medical Federated Learning",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="13--23",}
```



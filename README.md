## Install
git clone https://github.com/IAMJackYan/Fed-LWR.git
pip install -r requirements.txt

## Prepare data
1) Download the retinal fundus segmentation [dataset](https://drive.google.com/file/d/1p33nsWQaiZMAgsruDoJLyatoq5XAH-TH/view).
2) Use the script to preprocess the data
python preprocess_rif.py  # Please change the file to input your data path.

## Run
python Fedlwr.py

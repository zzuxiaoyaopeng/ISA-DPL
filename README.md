# Identity Suppression Awareness for Facial Expression Recognition via Dual-Path Learning
This repository is the implement of our paper Identity Suppression Awareness for Facial Expression Recognition via Dual-Path Learning.
## Model Architecture
![image](https://github.com/zzuxiaoyaopeng/ISA-DPL/blob/main/model_main_16.png)
## Setup
* Check the packages needed or simply run the command:
```
pip install -r requirements.txt
```
* Download the ISA-DPL checkpoints and the pre-trained checkpoints form [here](https://pan.baidu.com/s/1u16MkoM_PPXb7L5e_PsiVg?pwd=hm4a), and put them into vitfer/checkpoint/ and vitfer/prem_odels/.
## Train
* AffectNet:
```
python main_poster_affectnet_au_arbex_posterv2.py
```
* SFEW2.0
```
python main_poster_sfew_au_arbex_posterv2.py
```
* RAFDB
```
python main_poster_rafdb_au_arbex_posterv2.py
```
## Test
```
python confusion_main.py
```


  


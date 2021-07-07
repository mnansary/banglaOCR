# banglaOCR

```python
Version: 0.0.6     
Authors: Md. Nazmuddoha Ansary 
```

**LOCAL ENVIRONMENT**  
```python
OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver        
Memory      : 7.7 GiB  
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.28.2  
```
# Setup
>Assuming the **libraqm** complex layout is working properly, you can skip to **python requirements**. 
*  ```sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools```
* Install libraqm as described [here](https://github.com/HOST-Oman/libraqm)
* ```sudo ldconfig``` (librarqm local repo)

**python requirements**
* **pip requirements**: ```pip3 install -r requirements.txt``` 
> Its better to use a virtual environment 
OR use conda-
* **conda**: use environment.yml
* install **tesseract** if you want to use the **tesseract-ocr**. Make sure to properly setup the bangla data. 

# Pretraining Dataset
* The overall dataset is available here: https://www.kaggle.com/nazmuddhohaansary/sourcedata
* We only need the **bangla** folder for this
* The path for the **source** folder is the **data_path** used in **datagen_pretrain.py** and **datagen_finetune.py**
* The dataset is collected and compiled from vairous sources such as:
    * The bangla **grapheme** dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
        * Only the **256** folder under **256_train** is kept and renamed as **RAW** form **BengaliAI:Supplementary dataset for BengaliAI Competition**
    * The bangla **number** dataset is taken from [here](https://www.kaggle.com/nazmuddhohaansary/banglasymbols) 
        * Only the **RAW_NUMS** folder is kept that contains all the images of the numbers

# FineTuning DataSet

* For Modifier Finetuning the following open source dataset was used: https://www.kaggle.com/nazmuddhohaansary/boisebangladata
* This data set is variation of dataset introduced by:
[Nishatul Majid](https://orcid.org/0000-0001-5445-5252) and [Elisa Barney-Smith](https://orcid.org/0000-0003-2039-3844)
```[DOI](https://doi.org/10.18122/saipl/1/boisestate)```


* The folder structre should look as follows:

```python
        source
        ├── bangla
           ├── graphemes.csv
           ├── numbers.csv
           ├── dictionary.csv
           ├── fonts
           ├── graphemes
           └── numbers
        ├── boise_state
            
```
# Execution
* run **datagen_pretrain.py**
* run **datagen_finetune.py**
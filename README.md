# banglaOCR
Mostly contains grapheme 'image' representation code 

```python
Version: 0.0.1     
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
* ```pip3 install -r requirements.txt```

# Dataset
* The dataset is taken from [here](https://www.kaggle.com/pestipeti/bengali-quick-eda/#data). 
* Only the **256** folder under **256_train** is kept and renamed as **RAW** form **BengaliAI:Supplementary dataset for BengaliAI Competition**
* And the **train.csv** under **Bengali.AI Handwritten Grapheme Classification:Classify the components of handwritten Bengali** is renamed as **label.csv**
* The final **data** folder is like as follows:
```python
    data
    ├── label.csv
    └── RAW
```

# Entry Point
* run **data.py**
```python

    usage: data.py [-h] --data_path DATA_PATH

    script to create handwritten to printed text data from bengal.ai grapheme data

    optional arguments:
    -h, --help            show this help message and exit
    --data_path DATA_PATH
                            The path to the folder that contains label.csv and RAW
                            folder (default: None)

```
* after execution the folder should have **images** and **targets** as follows:
```python
    data
    ├── images
    ├── targets
    ├── label.csv*
    └── RAW
```
* run **main.py** with the same **data_path** as **data.py**
```python

    usage: main.py [-h] --data_path DATA_PATH

    script to create handwritten to printed text data from bengal.ai grapheme data

    optional arguments:
    -h, --help            show this help message and exit
    --data_path DATA_PATH
                            The path to the folder that contains label.csv and RAW
                            folder (default: None)

```


# NOTES
* **TODO:** 
    *   Add error handling
    *   Documentation


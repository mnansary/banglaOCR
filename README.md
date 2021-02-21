# banglaOCR
Mostly contains grapheme 'image' representation code 

```python
Version: 0.0.2     
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
* create the specific dataset such as: **grapheme** or **word** with corresponding script
* run **main.py**
```python

    usage: main.py [-h] --data_dir DATA_DIR

    script to create handwritten to printed text data from bengal.ai grapheme data

    optional arguments:
    -h, --help           show this help message and exit
    --data_dir DATA_DIR  The path to the folder that contains images and targets
                        (default: None)

```


# NOTES
* **TODO:** 
    *   Add error handling
    *   Documentation


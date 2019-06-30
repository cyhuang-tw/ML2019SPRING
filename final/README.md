# Machine Learning Final Project
## News Retrieval – AI CUP 2019 Competition

### Prerequisites
* python 3.6
* gensim==3.7.2
* jieba==0.39
* numpy==1.16.4
* pandas==0.24.2
* scikit-learn==0.21.0
* tqdm==4.19.9

### Usage
```
cd src
bash run.sh <dict_file> <json_file> <TD_file> <NC_file> <query_file> <output_file>
```
* dict_file: dictionary file required for jieba to perform word segmentation.
* json_file: json file containing news content.
* TD_file: TD.csv provided by the organizer.
* NC_file: NC_1.csv provided by the organizer.
* query_file: QS_1.csv provided by the organizer.
* output_file: output file (.csv) containing the ranking result.
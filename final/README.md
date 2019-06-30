# Machine Learning Final Project
## News Retrieval – AI CUP 2019 Competition
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
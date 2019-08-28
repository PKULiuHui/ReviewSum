# ReviewSum

Codes and datasets for our paper: *Neural Review Summarization Leveraging User and Product Information* (CIKM19)

## Requirements

* python >= 3.5
* pytorch >= 1.1.0
* sumeval
* tqdm

## Datasets

Download the datasets from [Google Drive](https://drive.google.com/open?id=1-5LAp4cw5jwtj3J8GRtHtkvYlzo7iis5) or [Baidu Pan](https://pan.baidu.com/s/1UusN1_LHzj6ObOWw5BskIA). Put the unziped /data directory into the project root directory.

## Models

The repository contains three baseline models: **seq2seq**, **seq2seqAttn**, **pgn**, and our proposed four models: **AttrEnc**, **AttrDec**, **AttrEncDec**, and **MemAttr**, as mentioned in our paper. The command to run each model is the same. Take our novel model **MemAttr** as an example:

Train a **MemAttr** model:

```
cd code/memAttr
python train.py 
```

Test the trained model:

```
python train.py -test -load_model <the_checkpoint_you_want_to_test>
```

The train and test parameters can be found in the source code train.py.

## Citation

If you use our codes or datasets in your research, please kindly cite our paper: *Neural Review Summarization Leveraging User and Product Information* (CIKM19)



 
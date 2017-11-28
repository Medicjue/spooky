Spooky Author Identification Kaggle Competition
-----------------------------------------------
11/14
Baseline Program
1. Using Random Forest
2. Separate Train Data as 0.7 Training / 0.3 Testing
3. Convert text as one-hot encoding vectors
4. Using JIEBA to tokenize text
Use 0.7 train data for model training
Result: 1.36856

11/15
Baseline w/ all train data
Still using Random Forest
Result: 1.31580

11/15
Removing stopwords, removing marks
Getting worse
Result: 1.38007

11/15
Using SVM(Linear)
Result: 0.60411

11/18
Download fasttext pretrain word vectors
https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
Using k-means to cluster vectors into 20 groups
Convert text into histogram vectors
Then use SVM as classifier
Result: 1.05575

11/18
k-means into 40 groups
Result: 1.04287

11/21
Apply fastText
Result: 2.06847



11/28
LSTM 1 epoch
NTLK
Result: 2.42336
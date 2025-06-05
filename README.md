# MPT
This is the repository of paper: Multimodal Sentiment Analysis with Multi-Perspective Thinking via Large Multimodal Models (Targeted Submission to CIKM 2025)

![image](https://github.com/user-attachments/assets/0fb7c4a2-0565-49d1-a01f-e91b6c0a62d3)

# Requrements
We give the version of the python package we used.
```python
python 3.8
pytorch 1.12.0
```

# Data
In this paper, we construct four datasets with multi-perspective thinkings from the LMMs in multimodal sentiment analysis tasks on the basis of the datasets MVSA, Memotion and the CH-Mits. These four datasets file is given in this repository whose filename ends with '.tsv'.

We upload the original MVSA-Single dataset in the MVSA-Single folder.

For original MVSA-Multiple dataset, please refer to https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

For original Memotion dataset, please refer to paper "Task Report: Memotion Analysis 1.0 @SemEval 2020: The Visuo-Lingual Metaphor!"

For original CH-Mits dataset, please refer to https://github.com/Marblrdumdore/CH-Mits

# Run Code
Before running the code, you should download the pretrained parameters of the textual encoder and the visual encoder. Due to the large size of the pretrained models, we give the downloading links so that you can access them.

For the textual encoder, you can refer to the open-source BERT model through https://huggingface.co/google-bert/bert-base-uncased

For the visual encoder, you can access the ViT model through https://huggingface.co/google/vit-base-patch16-224-in21k.

After getting the pre-trained models mentioned above, you can put the textual encoder in the folder ```bert/bert-base/``` and the visual encoder in the folder ```models/weight/```.

Then we can seamlessly run the code:

```python
python main.py
```

If you want to change the dataset, you can modify the dataset path in the lines 10 and 135 in the file ```data.py```

# GenAI Disclosure
There was no use of GenAI toos whatsoever in any stage of the research.


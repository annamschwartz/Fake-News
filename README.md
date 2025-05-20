# Fake News Classifier

## Introduction & Background:

Is it possible to determine whether a news article is real or fake based on its title and contents? 

How disinformation news outlets use phrasing, grammar, capitalization, and punctuation to bait people into reading fake news

Real: U.S. appeals court hears arguments on Virginia's voter ID law

Fake: FAIL! The Trump Organization’s Credit Score Will Make You Laugh

Want to see how easily fake news can be identified and how easy it could be for disinformation website to bait people into reading fake news

“Research finds that the ordinary people have a hard time identifying false news... We tend to be overconfident in our ability to distinguish ‘real’ from fake when it comes to online news information. This overconfidence leads to an increased chance of sharing fake news on social media platforms.” [3]

We predict that we will be able to train a model to identify real vs. fake news

## Experimental Setup:

Dataset:
- Sets of confirmed fake and confirmed real news articles from Hugging Face as well as datasets we can creta on our own to really test

Model
- Roberta-Based Fine-Tuned model found on Hugging Face, train the model with data sets, tune to what we want, test with other datasets 

Evaluation metrics:
- We will evaluate by testing it on articles not in the datasets and see if our model evaluates it as real or fake. It will be successful if our model is able to correctly identify an article as real or fake based on its title and contents. If we get negative results we will conclude that it is difficult to determine real news from fake news and figuring out that information comes down to doing additional research and we cannot just determine from the content within the article 

## Results:

|           | True   | False  | Total   |
|-----------|--------|--------|---------|
| **True**  | 12,778 | 173    | 12,951  |
| **False** | 253    | 10,859 | 11,112  |
| **Total** | 13,031 | 11,032 | 24,063  |

98.22% accuracy
98.66% precision True
97.72% precision False
98.06% Recall True
98.43% Recall False

Real Title: “Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest”

The model predicts it correctly, but how does certain words change the score?

Adding “LOL” and “OMG” to the title can cause the score of it being real to decrease and the score of it being fake it increase. This can be what causes the ones that are falsely identified as false even though they are true to be classified wrong.  An exclamation mark at the end of the sentence of just the title had the most impact on the  decreased of the real score and the increased of the fake score. 

Ones that were wrongly classified as True were more political.

Titles with Trump in the title were more likely to be classified as False.

## Summary & Conclusion:

It is possible to determine whether a news article is real or fake based on its title and contents and our model can predict it with a 98% accuracy. 

It was interesting to look at the different words, punctuation, and phrases that would skew our results to be more likely classified as either True or False. 

In general, we got the results we expected, but we were able to look more into the false positives and false negatives identified and why we think they were classified that way. 

Some limitations we had was running the code took a long time due to the torch softmax to create the final probability. So we had to use a smaller data set to test the code. Other limitations include finding datasets that worked with our model.

## References:

1. GonzaloA. (2021, December 16). *Fake news*. Hugging Face.  
   https://huggingface.co/datasets/GonzaloA/fake_news

2. hamzab. (2022, March 29). *Roberta-fake-news-classification*. Hugging Face.  
   https://huggingface.co/hamzab/roberta-fake-news-classification

3. *How is Fake News Spread? Bots, People like You, Trolls, and Microtargeting.* (n.d.). Center for Information Technology & Society.  
   https://cits.ucsb.edu/fake-news/spread#spread-people

4. koliskos. (2023, March 12). *Fake news*. Hugging Face.  
   https://huggingface.co/datasets/koliskos/fake_news

5. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (arXiv:1907.11692). arXiv.  
   https://doi.org/10.48550/arXiv.1907.11692

<i>Note: This project was completed in Fall 2023 in Professor Davis's NLP course at Colgate University. It was completed by Anna Schwartz and Ellie Humphreys</i>

Final Poster: [https://docs.google.com/presentation/d/1GE7kR9-fzJQkHfOVhsbRYqk7Sied50grlyCim0kftUY/edit?usp=sharing](url)

Model: [https://huggingface.co/hamzab/roberta-fake-news-classification?text=Some+ninja+attacked+the+White+House](url)

Datasets: [https://drive.google.com/drive/u/1/folders/1ImehAAJNL76U6iFbZkft-C_yCqMiIfJS](url)

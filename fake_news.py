
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

def predict_fake(title,text):
    input_str = "<title>" + title + "<content>" +  text + "<end>"
    input_ids = tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    device =  'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    return dict(zip(["Fake","Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ))
    
#print(predict_fake(<HEADLINE-HERE>,<CONTENT-HERE>))
#print(model)

usecol = ["title","text"]
true_data = pd.read_csv("True.csv", usecols=usecol)
fake_data = pd.read_csv("Fake.csv", usecols=usecol)


real_true = 0
fake_true = 0
real_fake = 0
fake_fake = 0
'''
for index, row in true_data.iterrows():
    title = row["title"]
    text = row["text"]
    predict = predict_fake(title,text)
    if predict['Fake']>predict['Real']:
        fake_fake+=1
    else:
        real_true+=1

for index, row in fake_data.iterrows():
    title = row["title"]
    text = row["text"]
    predict = predict_fake(title,text)
    if predict['Fake']>predict['Real']:
        real_fake+=1
    else:
        fake_true+=1
'''

cols = ["","title","text","label"]
test = pd.read_csv("train.csv", sep=';')
true_dic = {}
false_dic = {}
for index, row in test.iterrows():
    title = row["title"]
    text = row["text"]
    label = row["label"]
    if label == 1:
        true_dic[title] = [text]
    else:
        false_dic[title] = [text]

#print(true_dic)
#print(false_dic)

real_true = 0
fake_true = 0
real_fake = 0
fake_fake = 0
wrong_true = []
wrong_false = []

for key, val in true_dic.items():
    val = val[0]
    predict = predict_fake(key,val)
    if predict['Real']>predict['Fake']:
        real_true+=1
    else:
        fake_fake+=1
        wrong_false.append(key)

for key, val in false_dic.items():
    val = val[0]
    predict = predict_fake(key,val)
    if predict['Fake']>predict['Real']:
        real_fake+=1
    else:
        fake_true+=1
        wrong_true.append(key)

print("Real True:", real_true)
print("Fake True:", fake_true)
print("Real Fake:", real_fake)
print("Fake Fake:", fake_fake)

print("Wrong False:", wrong_false)
print("Wrong True:", wrong_true)

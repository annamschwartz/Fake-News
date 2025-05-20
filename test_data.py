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
    

title = "Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest"
text = "RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus  traditional birthplace in Bethlehem on Wednesday night in protest at U.S. President Donald Trump s decision to recognize Jerusalem as Israel s capital. A Christmas tree adorned with lights outside Bethlehem s Church of the Nativity, where Christians believe Jesus was born, and another in Ramallah, next to the burial site of former Palestinian leader Yasser Arafat, were plunged into darkness.  The Christmas tree was switched off on the order of the mayor today in protest at Trump s decision,  said Fady Ghattas, Bethlehem s municipal media officer.  He said it was unclear whether the illuminations would be turned on again before the main Christmas festivities. In a speech in Washington, Trump said he had decided to recognize Jerusalem as Israel s capital and move the U.S. embassy to the city. Israeli Prime Minister Benjamin Netanyahu said Trump s move  marked the beginning of a new approach to the Israeli-Palestinian conflict and said it was an  historic landmark . Arabs and Muslims across the Middle East condemned the U.S. decision, calling it an incendiary move in a volatile region and the European Union and United Nations also voiced alarm at the possible repercussions for any chances of reviving Israeli-Palestinian peacemaking."

predict = predict_fake(title,text)
print("Real score:", predict['Real'])
print("Fake score", predict["Fake"])
print()
title = "lol Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest"
text = "RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus  traditional birthplace in Bethlehem on Wednesday night in protest at U.S. President Donald Trump s decision to recognize Jerusalem as Israel s capital. A Christmas tree adorned with lights outside Bethlehem s Church of the Nativity, where Christians believe Jesus was born, and another in Ramallah, next to the burial site of former Palestinian leader Yasser Arafat, were plunged into darkness.  The Christmas tree was switched off on the order of the mayor today in protest at Trump s decision,  said Fady Ghattas, Bethlehem s municipal media officer.  He said it was unclear whether the illuminations would be turned on again before the main Christmas festivities. In a speech in Washington, Trump said he had decided to recognize Jerusalem as Israel s capital and move the U.S. embassy to the city. Israeli Prime Minister Benjamin Netanyahu said Trump s move  marked the beginning of a new approach to the Israeli-Palestinian conflict and said it was an  historic landmark . Arabs and Muslims across the Middle East condemned the U.S. decision, calling it an incendiary move in a volatile region and the European Union and United Nations also voiced alarm at the possible repercussions for any chances of reviving Israeli-Palestinian peacemaking."
predict1 = predict_fake(title,text)
print("Real LOL score:", predict1['Real'])
print("Fake LOL score", predict1["Fake"])
print()
title = "wow Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest"
text = "RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus  traditional birthplace in Bethlehem on Wednesday night in protest at U.S. President Donald Trump s decision to recognize Jerusalem as Israel s capital. A Christmas tree adorned with lights outside Bethlehem s Church of the Nativity, where Christians believe Jesus was born, and another in Ramallah, next to the burial site of former Palestinian leader Yasser Arafat, were plunged into darkness.  The Christmas tree was switched off on the order of the mayor today in protest at Trump s decision,  said Fady Ghattas, Bethlehem s municipal media officer.  He said it was unclear whether the illuminations would be turned on again before the main Christmas festivities. In a speech in Washington, Trump said he had decided to recognize Jerusalem as Israel s capital and move the U.S. embassy to the city. Israeli Prime Minister Benjamin Netanyahu said Trump s move  marked the beginning of a new approach to the Israeli-Palestinian conflict and said it was an  historic landmark . Arabs and Muslims across the Middle East condemned the U.S. decision, calling it an incendiary move in a volatile region and the European Union and United Nations also voiced alarm at the possible repercussions for any chances of reviving Israeli-Palestinian peacemaking."
predict2 = predict_fake(title,text)
print("Real WOW score:", predict2['Real'])
print("Fake WOW score", predict2["Fake"])
print()
title = "omg Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest"
text = "RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus  traditional birthplace in Bethlehem on Wednesday night in protest at U.S. President Donald Trump s decision to recognize Jerusalem as Israel s capital. A Christmas tree adorned with lights outside Bethlehem s Church of the Nativity, where Christians believe Jesus was born, and another in Ramallah, next to the burial site of former Palestinian leader Yasser Arafat, were plunged into darkness.  The Christmas tree was switched off on the order of the mayor today in protest at Trump s decision,  said Fady Ghattas, Bethlehem s municipal media officer.  He said it was unclear whether the illuminations would be turned on again before the main Christmas festivities. In a speech in Washington, Trump said he had decided to recognize Jerusalem as Israel s capital and move the U.S. embassy to the city. Israeli Prime Minister Benjamin Netanyahu said Trump s move  marked the beginning of a new approach to the Israeli-Palestinian conflict and said it was an  historic landmark . Arabs and Muslims across the Middle East condemned the U.S. decision, calling it an incendiary move in a volatile region and the European Union and United Nations also voiced alarm at the possible repercussions for any chances of reviving Israeli-Palestinian peacemaking."
predict1 = predict_fake(title,text)
print("Real OMG score:", predict1['Real'])
print("Fake OMG score", predict1["Fake"])
print()
title = "Palestinians switch off Christmas lights in Bethlehem in anti-Trump protest!"
text = "RAMALLAH, West Bank (Reuters) - Palestinians switched off Christmas lights at Jesus  traditional birthplace in Bethlehem on Wednesday night in protest at U.S. President Donald Trump s decision to recognize Jerusalem as Israel s capital. A Christmas tree adorned with lights outside Bethlehem s Church of the Nativity, where Christians believe Jesus was born, and another in Ramallah, next to the burial site of former Palestinian leader Yasser Arafat, were plunged into darkness.  The Christmas tree was switched off on the order of the mayor today in protest at Trump s decision,  said Fady Ghattas, Bethlehem s municipal media officer.  He said it was unclear whether the illuminations would be turned on again before the main Christmas festivities. In a speech in Washington, Trump said he had decided to recognize Jerusalem as Israel s capital and move the U.S. embassy to the city. Israeli Prime Minister Benjamin Netanyahu said Trump s move  marked the beginning of a new approach to the Israeli-Palestinian conflict and said it was an  historic landmark . Arabs and Muslims across the Middle East condemned the U.S. decision, calling it an incendiary move in a volatile region and the European Union and United Nations also voiced alarm at the possible repercussions for any chances of reviving Israeli-Palestinian peacemaking."
predict1 = predict_fake(title,text)
print("Real ! score:", predict1['Real'])
print("Fake ! score", predict1["Fake"])
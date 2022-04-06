import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import json

import re
import spacy
import string
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import time

nlp = spacy.load("en_core_web_sm")

table = str.maketrans('', '', string.punctuation)

stopwords = ["a", "about", "above", "after", "again", "against",
            "all", "am", "an", "and", "any", "are", "as", "at", "be"]

def preprocessing(line):
    print("input line: ", line)
    #for item in line:
    
    sentence = line.lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("_", " _ ")
    sentence = sentence.replace("/", " / ")
    print("After replace special characters: ", sentence)
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    print("Output line: ", filtered_sentence)
    return filtered_sentence

with open('intent_chatbot.json') as f:
    intents = json.load(f)
    

inputs, targets = [], []
classes = []
intent_doc = {}

for intent in intents['intents']:
    if intent['intent'] not in classes:
        classes.append(intent['intent'])
    if intent['intent'] not in intent_doc:
        intent_doc[intent['intent']] = []
    for text in intent['text']:
        print(text)
        inputs.append(preprocessing(text))
        targets.append(intent['intent'])
    for response in intent['responses']:
        intent_doc[intent['intent']].append(response)

def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(input_list)
    input_seq = tokenizer.texts_to_sequences(input_list)
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')
    return tokenizer, input_seq

tokenizer, input_tensor = tokenize_data(inputs)

def create_categorical_target(targets):
    word = {}
    categorical_target = []
    counter = 0
    for trg in targets:
        if trg not in word:
            word[trg] = counter
            counter+=1
        categorical_target.append(word[trg])
    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    reverse_word_index = {}
    for k, v in word.items():
        reverse_word_index[v] = k
    return categorical_tensor,  reverse_word_index

target_tensor, trg_index_word = create_categorical_target(targets)

epochs = 50
vocab_size = len(tokenizer.word_index) + 1
embed_dim = 512
units = 128
target_length = target_tensor.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size,embed_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
    tf.keras.layers.Dense(units, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(target_length, activation='softmax')   
])

optimizer = tf.keras.optimizers.Adam(lr=1e-2)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

history = model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])

def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))

    # slpit the input sentences into words
    for token in doc:
        if token.text in tokenizer.word_index:
            sent_seq.append(tokenizer.word_index[token.text])

        # handle the unknown words error
        else:
            sent_seq.append(tokenizer.word_index['<unk>'])
        
    sent_seq = tf.expand_dims(sent_seq, 0)

        # predict the category of input sentences
    pred = model(sent_seq)
    pred_class = np.argmax(pred.numpy(), axis=1)

        # choice a random response for predicted sentence
    return random.choice(intent_doc[trg_index_word[pred_class[0]]])

def plot_graphs(history, string):
  plt.plot(history.history[string])
  #plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

def response(sentence):
    sent_seq = []
    doc = nlp(repr(sentence))
    for token in doc:
        if token.text in tokenizer.word_index:
            print(token.text)
            sent_seq.append(tokenizer.word_index[token.text])
            sent_seq = tf.expand_dims(sent_seq, 0)
    pred = model.predict(sent_seq)
    pred_class = np.argmax(pred, axis=1)
    return random.choice(intent_doc[trg_index_word[pred_class[0]]])

# Selenium for Gmail automation 
username = "laxmareddybilla980@gmail.com"
password = "Automation#NLP"

url = "https://accounts.google.com/ServiceLogin?service=mail&passive=true&rm=false&continue=https://mail.google.com/mail/&ss=1&scc=1&ltmpl=default&ltmplcache=2&emr=1&osid=1#identifier"

chrome = "C:/ChromeDriver75/chromedriver.exe"

driver = webdriver.Chrome(chrome)

driver.get(url)
driver.maximize_window()


driver.find_element(By.ID, "identifierId").send_keys(username)
WebDriverWait(driver, 3)
driver.find_element(By.ID, "identifierNext").click()
time.sleep(3)
driver.find_element(By.NAME, "password").send_keys(password)
driver.find_element(By.ID, "passwordNext").click()
print("Login was successful!")
time.sleep(5)

username = "laxmareddybilla980@gmail.com"
password = "Automation#NLP"

url = "https://accounts.google.com/ServiceLogin?service=mail&passive=true&rm=false&continue=https://mail.google.com/mail/&ss=1&scc=1&ltmpl=default&ltmplcache=2&emr=1&osid=1#identifier"

chrome = "C:/ChromeDriver75/chromedriver.exe"

class GmailAutomation:
    def __init__(self, chrome):
        self.chrome = chrome
        self.driver = webdriver.Chrome(self.chrome)

    def browser_action(self, url):
        self.driver.get(url)
        self.driver.maximize_window()

    def gmail_login(self, username, password):
        self.driver.find_element(By.ID, "identifierId").send_keys(username)
        WebDriverWait(self.driver, 3)
        self.driver.find_element(By.ID, "identifierNext").click()
        time.sleep(3)
        self.driver.find_element(By.NAME, "password").send_keys(password)
        self.driver.find_element(By.ID, "passwordNext").click()
        print("Login was successful!")
        time.sleep(5)

    def mailcompose(self):
        time.sleep(2)
        print(self.driver.current_url)
        self.driver.find_element(By.XPATH,"//div[@class='aic']").click()
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//textarea[@name='to']").send_keys("billalaxman@gmail.com")
        self.driver.find_element(By.XPATH,"//input[@name='subjectbox']").send_keys("Gmail automation with NLP demo")
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//div[@class='Ar Au']//div").send_keys("Hello")
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//div[@class='dC']/div[contains(text(),'Send')]").click()
        print("Email has been sent successully")
        time.sleep(2)

    def emailreply(self):
        self.driver.find_element(By.NAME,"q").send_keys("billalaxman@gmail.com")
        time.sleep(2)
        self.driver.find_element(By.XPATH, "//*[@class='gssb_m']/tbody/tr[3]").click()
        time.sleep(2)
        input_ = input(self.driver.find_element(By.XPATH, "//div[@class='a3s aiL ']/div[1]").text)
        print("Input: ", input_)
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//div[@aria-label='Reply']").click()
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//div[@aria-label='Message Body']").send_keys(response(input_))
        time.sleep(2)
        self.driver.find_element(By.XPATH,"//div[@class='dC']/div[contains(text(),'Send')]").click()
        time.sleep(10)
        print("Email was replied successfully")

        
obj = GmailAutomation(chrome)
obj.browser_action(url)
obj.gmail_login(username, password)
obj.mailcompose()

# for auto reply uncomment below command
#obj.emailreply()
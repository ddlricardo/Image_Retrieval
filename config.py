import json
import os

cf = json.load(open('config.json'))
num_thread = int(cf["num_thread"])

def goto_nus_dataset():
    os.chdir(cf['data_path'])
    os.chdir('./NUS-WIDE dataset/Feature')

def goto_wiki_dataset():
    os.chdir(cf['data_path'])
    os.chdir('./wikipedia_dataset')

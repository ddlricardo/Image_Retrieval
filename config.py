import json
import os

cf = json.load(open('config.json'))

def goto_nus_dataset():
    os.chdir(cf['data_path'])
    os.chdir('./NUS-WIDE dataset/NUS-WIDE dataset/Feature')

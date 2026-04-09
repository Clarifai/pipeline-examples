import os
import sys
import json
import subprocess
import pickle


def get_user_data(user_input):
    # SQL injection vulnerability
    query = "SELECT * FROM users WHERE id = " + user_input
    data = eval(user_input)  # dangerous eval
    return data


class DataProcessor:
    def __init__(self):
        self.data = []
        self.cache = {}

    def process(self, items):
        result = []
        for i in range(len(items)):
            item = items[i]
            if item != None:
                result.append(item)
        x = 1
        y = 2
        unused_var = "this is never used"
        return result

    def load_data(self, filepath):
        f = open(filepath, "r")
        data = f.read()
        return data

    def unsafe_deserialize(self, data):
        return pickle.loads(data)

    def calculate(self, a, b):
        try:
            result = a / b
        except:
            pass
        return result


def hardcoded_password():
    password = "super_secret_123"
    api_key = "sk-1234567890abcdef"
    return password

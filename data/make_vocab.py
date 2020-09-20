import os
import sys
import json
import random
import string

sys.path.append('..')
random.seed(1231)

from misc import util


data_file = sys.argv[1]

with open(data_file) as f:
    data = json.load(f)

vocab_names = ['char', 'word', 'regex']
vocab = {}
for name in vocab_names:
    vocab[name] = set()

new_data = {}

for split in ['train', 'val', 'test']:
    new_data[split] = []

    for item in data[split]:

        regex = item['re']
        task = item['task']
        examples = item['examples']

        for c in regex:
            vocab['regex'].add(c)

        for w in task:
            vocab['word'].add(w)

        for example in examples:
            for c in example[0]:
                vocab['char'].add(c)
            for c in example[1]:
                vocab['char'].add(c)

vocab['char'].add('@')
vocab['word'].add('@')
vocab['regex'].add('@')

for c in string.ascii_lowercase:
    vocab['char'].add(c)
    vocab['regex'].add(c)

for name in vocab_names:
    vocab[name] = sorted(list(vocab[name]))
    print(name, len(vocab[name]))

with open('vocab.json', 'w') as f:
    json.dump(vocab, f)




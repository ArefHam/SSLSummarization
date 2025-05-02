import json
from config import CONFIG as conf
from ast import literal_eval as make_tuple

#train_file = 'data/train.json'
#dev_file = 'data/dev.json'
#test_file = 'data/test.json'
train_file = conf['train_file']
dev_file = conf['dev_file']
test_file = conf['test_file']
max_sent_len = 120
max_doc_len = 80

def read_data(filename, add_first_sentence, keep_single_sent):
    data = []
    with open(filename) as in_file:
        for line in in_file:
            line = line.strip()
            all_sentences = []
            if add_first_sentence:
                all_sentences = [['<startsent>']]
            count = len(all_sentences)
            for sentence in line.split('##SENT##'):
                #sentence = sentence.split()[:max_sent_len]
                sentence = sentence.split()
                if len(sentence) > 0:
                    all_sentences.append(sentence)
                count+=1
                if count == max_doc_len:
                    break
            if keep_single_sent or len(all_sentences) > 1:
                data.append(all_sentences)
    return data

#if __name__ == '__main__':
def get_train_dev_test_data(add_first_sentence = False, keep_single_sent=True,
                            ignore_train=False):
    train_data = None
    if not ignore_train:
        train_data = read_data(train_file, add_first_sentence, keep_single_sent)
    dev_data = read_data(dev_file, add_first_sentence, keep_single_sent)
    test_data = read_data(test_file, add_first_sentence, keep_single_sent)
    return train_data, dev_data, test_data
    #print(len(train_data))
    #print(train_data[0])

def read_oracle(file_name):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            oracle_tuple = make_tuple(line.split('\t')[0])
            if oracle_tuple is not None:
                oracle_tuple = list(oracle_tuple)
                new_oracle_tuple = [i for i in oracle_tuple if i < max_doc_len]
                oracle_tuple = new_oracle_tuple
            else:
                oracle_tuple = []
            target.append(oracle_tuple)
    return target

def read_target_txt(file_name, is_combine=True):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            line = line.strip()
            sentences = line.split('##SENT##')
            if is_combine:
                target.append('\n'.join(sentences))
            else:
                target.append(sentences)
    return target

def read_target_20_news(file_name):
    target = []
    with open(file_name) as in_file:
        for line in in_file:
            line = line.strip()
            target.append(int(line))
    return target

import csv
from collections import defaultdict, deque

def load_dialogs_from_csv(csv_path):
    """
    Loads dialogs from a Twitter CSV file, chaining tweets using both in_response_to_tweet_id and response_tweet_id.
    Returns: List[List[str]] where each inner list is a dialog (list of utterances).
    """
    # Step 1: Parse CSV and build tweet dict
    tweets = {}
    children = defaultdict(list)
    roots = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = row["tweet_id"]
            tweets[tweet_id] = row
            in_response_to = row.get("in_response_to_tweet_id", "")
            if in_response_to:
                children[in_response_to].append(tweet_id)
            else:
                roots.add(tweet_id)
            # Also handle response_tweet_id (may be comma-separated list)
            response_ids = row.get("response_tweet_id", "")
            if response_ids:
                for resp_id in response_ids.split(","):
                    resp_id = resp_id.strip()
                    if resp_id:
                        children[tweet_id].append(resp_id)

    # Step 2: Find all roots (tweets not replying to anyone)
    for tweet_id in tweets:
        in_response_to = tweets[tweet_id].get("in_response_to_tweet_id", "")
        if in_response_to and in_response_to in tweets:
            roots.discard(tweet_id)

    # Step 3: Traverse each dialog tree (BFS for each root)
    dialogs = []
    visited = set()
    for root_id in roots:
        if root_id in visited:
            continue
        queue = deque()
        queue.append((root_id, []))
        while queue:
            current_id, path = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            current_row = tweets[current_id]
            current_text = current_row["text"]
            new_path = path + [current_text]
            # If no children, this is a leaf/dialog end
            if not children[current_id]:
                dialogs.append(new_path)
            else:
                for child_id in children[current_id]:
                    if child_id in tweets:
                        queue.append((child_id, new_path))
    # Optionally, sort utterances in each dialog by timestamp
    # (Assumes all tweets in a dialog are in correct order by traversal)
    return dialogs

import random

def split_customer_support(dialogs, dev_ratio=0.1, test_ratio=0.1, seed=1):
    """
    Given a list of dialogs, randomly shuffle (with `seed`) and
    return train/dev/test splits according to the given ratios.
    """
    random.seed(seed)
    dialogs = dialogs[:]            # copy
    random.shuffle(dialogs)

    n = len(dialogs)
    n_dev  = int(n * dev_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_dev - n_test

    train = dialogs[:n_train]
    dev   = dialogs[n_train:n_train + n_dev]
    test  = dialogs[n_train + n_dev:]
    return train, dev, test

    
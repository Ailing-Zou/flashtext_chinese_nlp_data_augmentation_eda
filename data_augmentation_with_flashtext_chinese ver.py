# arguments to be parsed from command line
import argparse
from flashtext import KeywordProcessor
import numpy as np
import pandas as pd
import random
from random import shuffle
random.seed(1)
import re
import jieba

# ----arguments----
ap = argparse.ArgumentParser()
ap.add_argument("--input", default='今天天气好晴朗,处处好风光', type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", default=6, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", default=1, type=float, help="Synonym replacement")
ap.add_argument("--alpha_ri", default=0.3, type=float, help="percent of words in each sentence to be randomly inserted")
ap.add_argument("--alpha_rs", default=0.2, type=float, help="percent of words in each sentence to be randomly swap")
ap.add_argument("--p_rd", default=0.3, type=float, help="percent of words in each sentence to be randomly deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join

    output = join(dirname(args.input), 'eda_' + basename(args.input))


# number of augmented sentences to generate per original sentence

def get_only_chars(line):
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.replace(" ", "")
    clean_line = line
    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

def keyword_processor_generation():
    keyword_processor = KeywordProcessor()
    data = open('merge_syno.txt', 'r', encoding='utf-8').readlines()
    word = []
    sys_word = []
    for i in data:
        word_item, sys_word_item = i.strip('\n').split(' ')
        word.append(word_item)
        sys_word.append(sys_word_item.split())
    keyword_dict = dict(zip(word, sys_word))
    keyword_processor.add_keywords_from_dict(keyword_dict)
    return keyword_processor

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

def synonym_replacement(words, n_sr):
    global new_sentence
    keyword_processor = keyword_processor_generation()
    new_words = words
    random_index = np.random.randint(0, len(new_words) - 1, size=n_sr)
    for i in random_index:
        words_item = new_words[i]
        new_words_item = keyword_processor.replace_keywords(words_item)
        new_sentence = new_words.replace(new_words[i], new_words_item)
    # print('sentence from Synonym Replacement:',new_sentence)
    return new_sentence


########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n_ri):
    new_words = words
    keyword_processor = keyword_processor_generation()
    random_synonym = keyword_processor.extract_keywords(words)
    random_syn_idx = np.random.randint(0, len(random_synonym) - 1, size=n_ri)
    for i in random_syn_idx:
        random_idx = np.random.randint(0, len(new_words) - 1)
        new_words = list(new_words)
        new_words.insert(random_idx, random_synonym[i])

    new_sentence = ''.join(new_words)
    # print('sentence from Random Insertion:', new_sentence)
    return new_sentence


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]

    # print('sentence from Random Swap :', ''.join(new_words))
    return new_words


########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    # obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    # print('sentence from Random Deletion:', ''.join(new_words))

    return new_words


########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    sentence = get_only_chars(sentence)  # clean all noise
    seg_list = jieba.cut(sentence, cut_all=False)
    jieba_result = " ".join(seg_list)
    words = jieba_result.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(''.join(words), n_sr)
        augmented_sentences.append(a_words)

    # ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(''.join(words), n_ri)
        augmented_sentences.append(a_words)

    # rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    # rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    # augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    # trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(sentence)

    return augmented_sentences


def gen_eda_sec_ver(lines, output_file, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug):
    aug_sentences = eda(lines, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug=num_aug)
    return aug_sentences

# main function
if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    # gen_eda(args.input, output, args.alpha_rs,args.p_rd, args.num_aug)
    modified_sentence = gen_eda_sec_ver(args.input, output, args.alpha_sr,
                                        args.alpha_ri, args.alpha_rs, args.p_rd, args.num_aug)

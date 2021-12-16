import os

in_domain_path = "data-bin/News"
out_domain_path = "data-bin/Thesis"

# get language
file_names = os.listdir(in_domain_path)
test_file_names = [name for name in file_names if 'test' in name and 'bin' in name]
assert test_file_names.__len__() == 2
s_lang = test_file_names[0].split('.')[1].split('-')[0]
t_lang = test_file_names[0].split('.')[1].split('-')[1]

# init new vocabs dict
s_key_vocabs = {}
t_key_vocabs = {}

# get map_index
for lang in (s_lang, t_lang):
    with open(os.path.join(in_domain_path, 'dict.'+lang+'.txt'), encoding='utf-8') as in_domain_dict_file:
        with open(os.path.join(out_domain_path, 'dict.'+lang+'.txt'), encoding='utf-8') as out_domain_dict_file:
            in_domain_word2idx = {}
            out_domain_word2idx = {}
            out_domain_idx2words = []
            for idx, line in enumerate(in_domain_dict_file.read().split('\n')):
                if line != '':
                    in_domain_word2idx[line.split(' ')[0]] = idx
            for idx, line in enumerate(out_domain_dict_file.read().split('\n')):
                if line != '':
                    out_domain_word2idx[line.split(' ')[0]] = idx
                    out_domain_idx2words.append(line.split(' ')[0])
            map_index = []
            for idx in range(len(out_domain_idx2words)):
                word = out_domain_idx2words[idx]
                if word in in_domain_word2idx:
                    map_index.append(in_domain_word2idx[word])
                else:
                    map_index.append(-1)
                    if lang == s_lang:
                        s_key_vocabs[word] = None
                    else:
                        t_key_vocabs[word] = None
            with open(os.path.join(out_domain_path, 'in2out_map_index.' + lang), 'w', encoding='utf-8') as out_file:
                out_file.write('\n'.join([str(item) for item in map_index]))
                
print("Key source words num:")
print(len(s_key_vocabs.keys()))
print("Key target words num:")
print(len(t_key_vocabs.keys()))

# print(t_key_vocabs)

# get paired sentences path
in_domain_sentences_path = "examples/translation/" + in_domain_path.split('/')[-1]
out_domain_sentences_path = "examples/translation/" + out_domain_path.split('/')[-1]
s_sentences = []
t_sentences = []
for sentences_path in (in_domain_sentences_path, out_domain_sentences_path):
    s_train_file_path = os.path.join(sentences_path, 'train.' + s_lang)
    t_train_file_path = os.path.join(sentences_path, 'train.' + t_lang)
    with open(s_train_file_path, encoding = 'utf-8') as sentences_file:
        s_sentences = sentences_file.read().split('\n')
    with open(t_train_file_path, encoding = 'utf-8') as sentences_file:
        t_sentences = sentences_file.read().split('\n')

# find the key data
has_added = {}
key_s_sentences = []
key_t_sentences = []

for idx, sentence in enumerate(s_sentences):  # source language
    if sentence == "":
        continue
    flag = False
    for word in sentence.split(' '):
        if word in s_key_vocabs:
            flag = True
            break
    if flag:
        key_s_sentences.append(s_sentences[idx])
        key_t_sentences.append(t_sentences[idx])
        has_added[idx] = None

for idx, sentence in enumerate(t_sentences):  # target language
    if sentence == "" or idx in has_added:
        continue
    flag = False
    for word in sentence.split(' '):
        if word in t_key_vocabs:
            flag = True
            break
    if flag:
        key_s_sentences.append(s_sentences[idx])
        key_t_sentences.append(t_sentences[idx])
    
# output key data
key_sentences_path = "examples/translation/" + in_domain_path.split('/')[-1] + '2' + out_domain_path.split('/')[-1] + "_key_data"
if not os.path.exists(key_sentences_path):
    os.makedirs(key_sentences_path)
for lang in (s_lang, t_lang):
    with open(os.path.join(key_sentences_path, 'train.' + lang), 'w', encoding = 'utf-8') as out_file:
        if lang == s_lang:
            out_file.write('\n'.join(key_s_sentences))
        else:
            out_file.write('\n'.join(key_t_sentences))

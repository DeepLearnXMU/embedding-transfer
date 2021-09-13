import os

in_domain_path = "data-bin/News"
out_domain_path = "data-bin/Thesis"

# get language
file_names = os.listdir(in_domain_path)
test_file_names = [name for name in file_names if 'test' in name and 'bin' in name]
assert test_file_names.__len__() == 2
s_lang = test_file_names[0].split('.')[1].split('-')[0]
t_lang = test_file_names[0].split('.')[1].split('-')[1]

#get map_index
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
            with open(os.path.join(out_domain_path, 'in2out_map_index.' + lang), 'w', encoding='utf-8') as out_file:
                out_file.write('\n'.join([str(item) for item in map_index]))
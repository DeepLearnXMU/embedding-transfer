#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import random
import sys
import copy
import time

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from examples.translation.subword_nmt import bpe_pack
from multiprocessing import Process, Queue
# from numba import jit
# import numba
# from numba.typed import Dict, List
# import multiprocessing


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(args):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.batch_size
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

def find_chld(word, word_dict, bpe_pack):
    res_string = bpe_pack.cut_word(word)
    if '@@' not in word and len(res_string.split(' ')) != 1:
        res = res_string.split(' ')
        res_idx = [word_dict.index(item) for item in res]
        return res, res_idx
    elif res_string.endswith('@@@ @') and len(res_string.split(' ')) != 3:
        res = res_string.split(' ')[:-1]
        res_idx = [word_dict.index(item) for item in res]
        return res, res_idx
    else:
        return my_find_chld(word, word_dict)

def my_find_chld(word, word_dict):
    # child
    cld_res = []
    cld_res_idx = []
    index = 0
    while index != word.__len__():
        flag = 0
        cp_back_index = word.__len__()
        for back_index in range(word.__len__(), index + 1, -1):
            if back_index == word.__len__() and word[index: back_index] in word_dict and word[index: back_index] != word: # 这里去除了自己，防止找到自身
                flag = 1
                cp_back_index = back_index
                break
            elif back_index != word.__len__() and word[index: back_index]+'@@' in word_dict and word[index: back_index]+'@@' != word: # 这里去除了自己，防止找到自身
                flag = 2
                cp_back_index = back_index
                break
        if flag == 1:
            cld_res.append(word[index: cp_back_index])
            cld_res_idx.append(word_dict.index(cld_res[-1]))
        elif flag == 2:
            cld_res.append(word[index: cp_back_index] + '@@')
            cld_res_idx.append(word_dict.index(cld_res[-1]))
        index = cp_back_index
    if '@@' in word:
        cld_res.append('@@@')
        cld_res_idx.append(word_dict.index(cld_res[-1]))
    return cld_res, cld_res_idx

# @njit(parallel=False)
# @jit(nopython=True)
def find_prt(word, symbols_list, symbols_index, max_len = 10):
    # parent
    prt_res = []
    prt_res_idx = []
    for idx, dict_symbol in enumerate(symbols_list):
        if len(prt_res_idx) >= max_len:
            break
        if word.replace('@@', '') in dict_symbol and word != dict_symbol: # 这里去除了自己，防止找到自身
            if ('@@' in word and not dict_symbol.endswith(word.replace('@@', ''))) or ('@@' not in word and dict_symbol.endswith(word)):
                prt_res.append(dict_symbol)
                prt_res_idx.append(symbols_index[dict_symbol])

    return prt_res, prt_res_idx

def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False

# def process(new_word, src_dict, pack, max_sub_position):
#     # print(new_word)
#     # print(src_dict)
#     # print(pack)
#     # exit()
#     chlds, chlds_idx = find_chld(new_word, src_dict, pack)
#     prts, prts_idx = find_prt(new_word, src_dict, max_len=10-len(chlds_idx))
#     now_len = len(chlds_idx) + len(prts_idx)
#     # if now_len > max_token_lens:
#     #     max_token_lens = now_len
#     tokens = chlds_idx + prts_idx
#     types = [1] * len(chlds_idx) + [2] * len(prts_idx)
#     position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
#     # print(tokens)

#     return (tokens, types, position)

def sample_worker(input_queue, output_queue, src_dict, tgt_dict):
    need_partition_rate = 0.4
    partition_rate = 0.2
    merge_rate = 0.5
    # need_partition_rate = 0
    # partition_rate = 0
    # merge_rate = 0
    max_sub_position = 98
    pack = bpe_pack.Bpepack('/user/fairseq/examples/translation/News_orig/tmp/codes')

    while(1):
        if input_queue.empty():
            time.sleep(1)
            continue
        try:
            samples = input_queue.get()
        except:
            continue
        if samples == 'close':
            del pack
            break
        for sample_idx in range(len(samples)):
            # Encoder Input
            sentence_len = samples[sample_idx]['net_input']['src_tokens'].size(1)
            total_gather_indexs = []
            total_chld_prt_tokens = []
            total_types = []
            total_positions = []
            max_nums = 1
            max_token_lens = 1
            max_gather_index_lens = 1
            padding_idx = 99999
            for line_idx, line_tokens in enumerate(samples[sample_idx]['net_input']['src_tokens']):
                # print([word_idx for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)])
                words = [src_dict[word_idx] for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==src_dict.eos_index or word_idx==src_dict.pad_index)]
                gather_index_helper = [i for i in range(0, sentence_len)]
                partition_words = []
                merge_words = []

                #todo 拆开与合并不要互相影响
                # Partition
                for idx, word in enumerate(words):
                    # if gather_index_helper[idx] > sentence_len:
                    #     continue

                    if word.replace('@@', '').__len__() > 6 and random.random() < need_partition_rate: # 先确定要拆成多少个token，随机选择位置，用负数的数值指示token个数，用于后续的整理
                        num_tokens = 1
                        init_idx = 0
                        for i in range(word.replace('@@', '').__len__() - 1):
                            now_rate = partition_rate
                            if i + 1 - init_idx == 1 or word.replace('@@', '').__len__() - (i + 1) == 1:
                                now_rate = partition_rate / 4
                                # now_rate = 0

                            if random.random() < now_rate and (word[init_idx:i+1] + '@@') not in src_dict:
                                num_tokens += 1
                                partition_words.append(word[init_idx:i+1] + '@@')
                                init_idx = i + 1
                        if init_idx != 0:
                            partition_words.append(word[init_idx:])
                            gather_index_helper[idx] = -num_tokens
                            for i in range(num_tokens):
                                gather_index_helper.append(len(gather_index_helper))

                # Merge
                idx = 0
                while idx < words.__len__():
                    if '@@' not in words[idx] or gather_index_helper[idx] < 0:
                        idx += 1
                        continue
                    now_idx = idx
                    while '@@' in words[now_idx] and random.random() < merge_rate and gather_index_helper[now_idx] > 0 and gather_index_helper[now_idx+1] > 0:
                        now_idx += 1
                    if now_idx == idx:
                        idx += 1
                        continue
                    gather_index_helper.append(len(gather_index_helper))
                    for i in range(idx, now_idx + 1):
                        if i == idx:
                            gather_index_helper[i] = gather_index_helper[-1]
                        else:
                            gather_index_helper[i] = '#'
                    if '@@' in words[now_idx]:
                        merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]])+'@@')
                    else:
                        merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]]))
                    idx = now_idx + 1
                
                # get gather index
                gather_index = []
                helper_index = sentence_len
                for idx in range(sentence_len):
                    helper = gather_index_helper[idx]
                    if helper == '#':
                        continue
                    if helper >= 0:
                        gather_index.append(helper)
                    else:
                        for i in range(-helper):
                            gather_index.append(gather_index_helper[helper_index])
                            helper_index += 1
                
                if len(gather_index) > max_gather_index_lens:
                    max_gather_index_lens = len(gather_index)
                total_gather_indexs.append(gather_index)

                this_chld_prt_tokens = []
                this_types = []
                this_positions = []
                now_nums = len(partition_words) + len(merge_words)
                if now_nums > max_nums:
                    max_nums = now_nums

                for new_word in (partition_words + merge_words):
                    chlds, chlds_idx = find_chld(new_word, src_dict, pack)
                    prts, prts_idx = find_prt(new_word, src_dict.symbols, src_dict.indices, max_len=10-len(chlds_idx))
                    # print(new_word)
                    # print(chlds)
                    # print(prts)
                    # exit()

                    now_len = len(chlds_idx) + len(prts_idx)
                    if now_len > max_token_lens:
                        max_token_lens = now_len
                    tokens = chlds_idx + prts_idx
                    if len(tokens) == 0:
                        tokens = [src_dict.pad_index]
                    position = []
                    for sub in chlds:
                        if new_word.startswith(sub):
                            position.append(0)
                        elif new_word.endswith(sub.replace('##', '')):
                            position.append(1)
                        else:
                            position.append(2)
                    for prt in prts:
                        if prt.startswith(new_word):
                            position.append(3)
                        elif prt.endswith(new_word.replace('##', '')):
                            position.append(4)
                        else:
                            position.append(5)
                    types = [1] * len(chlds_idx) + [2] * len(prts_idx)
                    # position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
                    this_chld_prt_tokens.append(tokens)
                    this_types.append(types)
                    this_positions.append(position)

                total_chld_prt_tokens.append(this_chld_prt_tokens)
                # total_chld_prt_tokens.append(this_chld_prt_tokens)
                total_types.append(this_types)
                total_positions.append(this_positions)

            # Add paddings
            token_padding_index = src_dict.pad_index
            position_padding_index = 6
            type_padding_index = 3
            for gather_index in total_gather_indexs:
                gather_index += ([padding_idx] * (max_gather_index_lens - len(gather_index)))
            for line_chld_prt_tokens in total_chld_prt_tokens:
                for single_tokens in line_chld_prt_tokens:
                    single_tokens += ([token_padding_index] * (max_token_lens - len(single_tokens)))
                line_chld_prt_tokens += ([[token_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_chld_prt_tokens))])
            for line_types in total_types:
                for single_tokens in line_types:
                    single_tokens += ([type_padding_index] * (max_token_lens - len(single_tokens)))
                line_types += ([[type_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_types))])
            for line_positions in total_positions:
                for single_tokens in line_positions:
                    single_tokens += ([position_padding_index] * (max_token_lens - len(single_tokens)))
                line_positions += ([[position_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_positions))])

            total_gather_indexs = torch.LongTensor(total_gather_indexs)
            total_chld_prt_tokens = torch.LongTensor(total_chld_prt_tokens)
            total_types = torch.LongTensor(total_types)
            total_positions = torch.LongTensor(total_positions)

            samples[sample_idx]['net_input']['gather_index'] = total_gather_indexs
            samples[sample_idx]['net_input']['chld_prt_tokens'] = total_chld_prt_tokens
            samples[sample_idx]['net_input']['types'] = total_types
            samples[sample_idx]['net_input']['positions'] = total_positions

            # Decoder Input
            sentence_len = samples[sample_idx]['net_input']['prev_output_tokens'].size(1)
            # total_gather_indexs = []
            temp_tgt_dict = copy.deepcopy(tgt_dict)
            max_num_target_tokens = 0 
            target_total_new_target_tokens = []
            target_total_chld_prt_tokens = []
            target_total_types = []
            target_total_positions = []
            max_nums = 1
            max_token_lens = 1
            # max_gather_index_lens = 1
            padding_idx = 99999
            for line_idx, line_tokens in enumerate(samples[sample_idx]['target']):
                # if line_idx == 0 or line_idx == 1: 
                #     continue
                # print([word_idx for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)])
                words = [tgt_dict[word_idx] for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==tgt_dict.eos_index or word_idx==tgt_dict.pad_index)]
                gather_index_helper = [i for i in range(0, sentence_len)]
                partition_words = []
                merge_words = []
                new_words = []

                #注意拆开与合并不要互相影响
                # Partition
                for idx, word in enumerate(words):
                    # if gather_index_helper[idx] > sentence_len:
                    #     continue

                    if word.replace('@@', '').__len__() > 2 and random.random() < need_partition_rate: # 先确定要拆成多少个token，随机选择位置，用负数的数值指示token个数，用于后续的整理
                        num_tokens = 1
                        init_idx = 0
                        for i in range(word.replace('@@', '').__len__() - 1):
                            now_rate = partition_rate
                            # if i + 1 - init_idx == 1 or word.replace('@@', '').__len__() - (i + 1) == 1:
                            #     now_rate = partition_rate / 4

                                # now_rate = 0

                            if random.random() < now_rate and (word[init_idx:i+1] + '@@') not in tgt_dict:
                                num_tokens += 1
                                partition_words.append(word[init_idx:i+1] + '@@')
                                init_idx = i + 1
                        if init_idx != 0:
                            partition_words.append(word[init_idx:])
                            gather_index_helper[idx] = -num_tokens
                            for i in range(num_tokens):
                                gather_index_helper.append(len(gather_index_helper))

                # Merge
                idx = 0
                while idx < words.__len__():
                    if '@@' not in words[idx] or gather_index_helper[idx] < 0:
                        idx += 1
                        continue
                    now_idx = idx
                    while '@@' in words[now_idx] and random.random() < merge_rate and gather_index_helper[now_idx] > 0 and gather_index_helper[now_idx+1] > 0:
                        now_idx += 1
                    if now_idx == idx:
                        idx += 1
                        continue
                    gather_index_helper.append(len(gather_index_helper))
                    for i in range(idx, now_idx + 1):
                        if i == idx:
                            gather_index_helper[i] = gather_index_helper[-1]
                        else:
                            gather_index_helper[i] = '#'
                    if '@@' in words[now_idx]:
                        merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]])+'@@')
                    else:
                        merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]]))
                    idx = now_idx + 1
                
                # get new target_tokens
                new_target_tokens = []
                helper_index = sentence_len
                merge_words_idx = 0
                partition_words_idx = 0
                for idx in range(sentence_len):
                    helper = gather_index_helper[idx]
                    if helper == '#':
                        continue
                    if helper >= 0 and helper < sentence_len: # normal
                        if int(line_tokens[idx]) != tgt_dict.eos_index and int(line_tokens[idx]) != tgt_dict.pad_index:
                            new_target_tokens.append(int(line_tokens[idx]))
                        # gather_index.append(helper)
                    elif helper >= sentence_len: # merged
                        new_word = merge_words[merge_words_idx]
                        merge_words_idx += 1
                        if new_word not in temp_tgt_dict:
                            new_words.append(new_word)
                        new_word_index = temp_tgt_dict.add_symbol(new_word)
                        # print(new_word)
                        new_target_tokens.append(new_word_index)
                    else: # separate
                        for i in range(-helper):
                            new_word = partition_words[partition_words_idx]
                            partition_words_idx += 1
                            # print(new_word)
                            if new_word not in temp_tgt_dict:
                                new_words.append(new_word)
                            new_word_index = temp_tgt_dict.add_symbol(new_word)
                            new_target_tokens.append(new_word_index)
                            # gather_index.append(gather_index_helper[helper_index])
                            helper_index += 1
                if len(new_target_tokens) > max_num_target_tokens:
                    max_num_target_tokens = len(new_target_tokens)
                # print(words)
                # print([temp_tgt_dict[idx] for idx in new_target_tokens])
                # exit()
                # if len(gather_index) > max_gather_index_lens:
                #     max_gather_index_lens = len(gather_index)
                # total_gather_indexs.append(gather_index)

                target_this_chld_prt_tokens = []
                target_this_types = []
                target_this_positions = []
                now_nums = len(partition_words) + len(merge_words)
                if now_nums > max_nums:
                    max_nums = now_nums

                for new_word in new_words:
                    chlds, chlds_idx = find_chld(new_word, tgt_dict, pack)
                    prts, prts_idx = find_prt(new_word, tgt_dict.symbols, tgt_dict.indices, max_len=10-len(chlds_idx))
                    # print(new_word)
                    # print(chlds)
                    # print(prts)
                    # exit()

                    now_len = len(chlds_idx) + len(prts_idx)
                    if now_len > max_token_lens:
                        max_token_lens = now_len
                    tokens = chlds_idx + prts_idx
                    if len(tokens) == 0:
                        tokens = [tgt_dict.pad_index]
                    position = []
                    for sub in chlds:
                        if new_word.startswith(sub):
                            position.append(0)
                        elif new_word.endswith(sub.replace('##', '')):
                            position.append(1)
                        else:
                            position.append(2)
                    for prt in prts:
                        if prt.startswith(new_word):
                            position.append(3)
                        elif prt.endswith(new_word.replace('##', '')):
                            position.append(4)
                        else:
                            position.append(5)
                    types = [1] * len(chlds_idx) + [2] * len(prts_idx)
                    # position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
                    target_this_chld_prt_tokens.append(tokens)
                    target_this_types.append(types)
                    target_this_positions.append(position)
                # print(new_target_tokens)
                target_total_new_target_tokens.append(new_target_tokens)
                target_total_chld_prt_tokens.append(target_this_chld_prt_tokens)
                # total_chld_prt_tokens.append(this_chld_prt_tokens)
                target_total_types.append(target_this_types)
                target_total_positions.append(target_this_positions)

            # Add paddings
            token_padding_index = tgt_dict.pad_index
            position_padding_index = 6
            type_padding_index = 3
            # for gather_index in total_gather_indexs:
            #     gather_index += ([padding_idx] * (max_gather_index_lens - len(gather_index)))
            new_prev_output_tokens = []
            new_target = []
            # print(target_total_new_target_tokens[-1])
            for idx in range(len(target_total_new_target_tokens)):
                # print(target_total_new_target_tokens[-1])
                temp1 = [tgt_dict.eos_index] + target_total_new_target_tokens[idx] + [tgt_dict.pad_index] * (max_num_target_tokens - len(target_total_new_target_tokens[idx]))
                temp2 = target_total_new_target_tokens[idx] + [tgt_dict.eos_index] + [tgt_dict.pad_index] * (max_num_target_tokens - len(target_total_new_target_tokens[idx]))
                # print(temp1)
                new_prev_output_tokens.append(temp1)
                new_target.append(temp2)
            # exit()
            # print(target_total_new_target_tokens)

            t_total_chld_prt_tokens = []
            for line_chld_prt_tokens in target_total_chld_prt_tokens:
                for single_tokens in line_chld_prt_tokens:
                    single_tokens += ([token_padding_index] * (max_token_lens - len(single_tokens)))
                t_total_chld_prt_tokens += line_chld_prt_tokens
                # line_chld_prt_tokens += ([[token_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_chld_prt_tokens))])
            t_total_types = []
            for line_types in target_total_types:
                for single_tokens in line_types:
                    single_tokens += ([type_padding_index] * (max_token_lens - len(single_tokens)))
                t_total_types += line_types
                # line_types += ([[type_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_types))])
            t_total_positions = []
            for line_positions in target_total_positions:
                for single_tokens in line_positions:
                    single_tokens += ([position_padding_index] * (max_token_lens - len(single_tokens)))
                t_total_positions += line_positions
                # line_positions += ([[position_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_positions))])
 
            # total_gather_indexs = torch.LongTensor(total_gather_indexs)
            target_total_chld_prt_tokens = torch.LongTensor(t_total_chld_prt_tokens)
            target_total_types = torch.LongTensor(t_total_types)
            target_total_positions = torch.LongTensor(t_total_positions)
            # print(new_prev_output_tokens[-1])
            # print(target_total_new_target_tokens)
            # print(new_prev_output_tokens.__len__())
            # print([len(item) for item in new_prev_output_tokens])
            new_prev_output_tokens = torch.LongTensor(new_prev_output_tokens)
            new_target = torch.LongTensor(new_target)

            # samples[sample_idx]['net_input']['gather_index'] = total_gather_indexs
            samples[sample_idx]['net_input']['target_chld_prt_tokens'] = target_total_chld_prt_tokens
            samples[sample_idx]['net_input']['target_types'] = target_total_types
            samples[sample_idx]['net_input']['target_positions'] = target_total_positions

            samples[sample_idx]['net_input']['prev_output_tokens'] = new_prev_output_tokens
            samples[sample_idx]['target'] = new_target
        
        # print(samples)
        # exit()
        output_queue.put(samples)


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum),
    )
    update_freq = (
        args.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args.update_freq)
        else args.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    # source_dict_dir = os.path.join(args.data, 'dict.' + args.source_lang + '.txt')
    # source_dict = {}
    # for idx, line in enumerate(open(source_dict_dir, encoding='utf-8').read().split('\n')):
    #     if line != '':
    #         source_dict[line.split(' ')[0]] = idx


    trainer.begin_epoch(epoch_itr.epoch)

    valid_losses = [None]
    valid_subsets = args.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    input_queue = Queue()
    output_queue = Queue()
    num_workers = 15
    processes = []
    src_dict = task.src_dict
    tgt_dict = task.tgt_dict
    for i in range(num_workers):
        print('Start processing')
        process = Process(target=sample_worker, args=(input_queue, output_queue, src_dict, tgt_dict))
        process.start()
        processes.append(process)
    # need_partition_rate = 0.4
    # partition_rate = 0.2
    # merge_rate = 0.5
    # # need_partition_rate = 0
    # # partition_rate = 0
    # # merge_rate = 0
    # max_sub_position = 98
    # pack = bpe_pack.Bpepack('/user/fairseq/examples/translation/News_orig/tmp/codes')

    # numba_dict = Dict.empty(
    #     key_type=numba.types.unicode_type,
    #     value_type=numba.types.int64,
    # )
    # numba_list = List()
    # for key in task.src_dict.indices.keys():
    #     numba_dict[key] = task.src_dict.indices[key]
    # for item in task.src_dict.symbols:
    #     numba_list.append(item)
    # for now_samples in progress:
    #     input_queue.put(now_samples)
    for i, now_samples in enumerate(progress):
        # print(now_samples)
        # exit()
        if i % 100 == 0:
            torch.cuda.empty_cache()
    # while(not input_queue.empty()):
        # print(samples)
        # print(task.src_dict)
        # exit()
        # sentence_batch_size = samples['src_tokens'].size(0)
        if i == 0:
            for j in range(len(processes)):
                input_queue.put(now_samples)
        while(output_queue.empty()):
            time.sleep(0.5)
            print(input_queue.qsize())
            pass
        samples = output_queue.get()
        input_queue.put(now_samples)

        # for sample_idx in range(len(samples)):
        #     sentence_len = samples[sample_idx]['net_input']['src_tokens'].size(1)
        #     total_gather_indexs = []
        #     total_chld_prt_tokens = []
        #     total_types = []
        #     total_positions = []
        #     max_nums = 1
        #     max_token_lens = 1
        #     max_gather_index_lens = 1
        #     padding_idx = 99999
        #     for line_idx, line_tokens in enumerate(samples[sample_idx]['net_input']['src_tokens']):
        #         # print([word_idx for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)])
        #         words = [task.src_dict[word_idx] for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)]
        #         gather_index_helper = [i for i in range(0, sentence_len)]
        #         partition_words = []
        #         merge_words = []

        #         #todo 拆开与合并不要互相影响
        #         # Partition
        #         for idx, word in enumerate(words):
        #             # if gather_index_helper[idx] > sentence_len:
        #             #     continue

        #             if word.replace('@@', '').__len__() > 6 and random.random() < need_partition_rate: # 先确定要拆成多少个token，随机选择位置，用负数的数值指示token个数，用于后续的整理
        #                 num_tokens = 1
        #                 init_idx = 0
        #                 for i in range(word.replace('@@', '').__len__() - 1):
        #                     now_rate = partition_rate
        #                     if i + 1 - init_idx == 1 or word.replace('@@', '').__len__() - (i + 1) == 1:
        #                         now_rate = partition_rate / 4
        #                         # now_rate = 0

        #                     if random.random() < now_rate and (word[init_idx:i+1] + '@@') not in task.src_dict:
        #                         num_tokens += 1
        #                         partition_words.append(word[init_idx:i+1] + '@@')
        #                         init_idx = i + 1
        #                 if init_idx != 0:
        #                     partition_words.append(word[init_idx:])
        #                     gather_index_helper[idx] = -num_tokens
        #                     for i in range(num_tokens):
        #                         gather_index_helper.append(len(gather_index_helper))

        #         # Merge
        #         idx = 0
        #         while idx < words.__len__():
        #             if '@@' not in words[idx] or gather_index_helper[idx] < 0:
        #                 idx += 1
        #                 continue
        #             now_idx = idx
        #             while '@@' in words[now_idx] and random.random() < merge_rate and gather_index_helper[idx] > 0:
        #                 now_idx += 1
        #             if now_idx == idx:
        #                 idx += 1
        #                 continue
        #             gather_index_helper.append(len(gather_index_helper))
        #             for i in range(idx, now_idx + 1):
        #                 if i == idx:
        #                     gather_index_helper[i] = gather_index_helper[-1]
        #                 else:
        #                     gather_index_helper[i] = '#'
        #             if '@@' in words[now_idx]:
        #                 merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]])+'@@')
        #             else:
        #                 merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]]))
        #             idx = now_idx + 1
                
        #         # get gather index
        #         gather_index = []
        #         helper_index = sentence_len
        #         for idx in range(sentence_len):
        #             helper = gather_index_helper[idx]
        #             if helper == '#':
        #                 continue
        #             if helper >= 0:
        #                 gather_index.append(helper)
        #             else:
        #                 for i in range(-helper):
        #                     gather_index.append(gather_index_helper[helper_index])
        #                     helper_index += 1
                
        #         if len(gather_index) > max_gather_index_lens:
        #             max_gather_index_lens = len(gather_index)
        #         total_gather_indexs.append(gather_index)

        #         this_chld_prt_tokens = []
        #         this_types = []
        #         this_positions = []
        #         now_nums = len(partition_words) + len(merge_words)
        #         if now_nums > max_nums:
        #             max_nums = now_nums
        #         # pool = multiprocessing.Pool(2)
        #         # words_length = len([partition_words + merge_words])
        #         # # print([partition_words + merge_words])
        #         # # exit()
        #         # pars = zip(partition_words + merge_words, [task.src_dict for j in range(words_length)], [pack for j in range(words_length)], [max_sub_position for j in range(words_length)])
        #         # res = pool.starmap(process, pars)
        #         # # print(res)
        #         # pool.close()
        #         # pool.join()
        #         # this_chld_prt_tokens = [item[0] for item in res]
        #         # this_types = [item[1] for item in res]
        #         # this_positions = [item[2] for item in res]
        #         # if len(res) != 0:
        #         #     now_len = max(len(item[0]) for item in res)
        #         #     if now_len > max_token_lens:
        #         #         max_token_lens = now_len
        #         for new_word in (partition_words + merge_words):
        #             chlds, chlds_idx = find_chld(new_word, task.src_dict, pack)
        #             prts, prts_idx = find_prt(new_word, task.src_dict.symbols, task.src_dict.indices, max_len=10-len(chlds_idx))
        #             now_len = len(chlds_idx) + len(prts_idx)
        #             if now_len > max_token_lens:
        #                 max_token_lens = now_len
        #             tokens = chlds_idx + prts_idx
        #             types = [1] * len(chlds_idx) + [2] * len(prts_idx)
        #             position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
        #             this_chld_prt_tokens.append(tokens)
        #             this_types.append(types)
        #             this_positions.append(position)
        #         total_chld_prt_tokens.append(this_chld_prt_tokens)
        #         # total_chld_prt_tokens.append(this_chld_prt_tokens)
        #         total_types.append(this_types)
        #         total_positions.append(this_positions)

        #         # print(gather_index)
        #         # print(words)
        #         # print(partition_words)
        #         # print(merge_words)
        #         # if len(partition_words) > 0:
        #         #     print(find_chld(partition_words[0], task.src_dict, pack) + find_prt(partition_words[0], task.src_dict))
        #         # if len(merge_words) > 0:
        #         #     print(find_chld(merge_words[0], task.src_dict, pack) + find_prt(merge_words[0], task.src_dict)) 
        #         # print('\n')
        #         # exit()
        #     total_chld_prt_tokens = list(total_chld_prt_tokens)
        #     # Add paddings
        #     token_padding_index = task.src_dict.pad_index
        #     position_padding_index = 99
        #     type_padding_index = 3
        #     for gather_index in total_gather_indexs:
        #         gather_index += ([padding_idx] * (max_gather_index_lens - len(gather_index)))
        #     for line_chld_prt_tokens in total_chld_prt_tokens:
        #         for single_tokens in line_chld_prt_tokens:
        #             single_tokens += ([token_padding_index] * (max_token_lens - len(single_tokens)))
        #         line_chld_prt_tokens += ([[token_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_chld_prt_tokens))])
        #     for line_types in total_types:
        #         for single_tokens in line_types:
        #             single_tokens += ([type_padding_index] * (max_token_lens - len(single_tokens)))
        #         line_types += ([[type_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_types))])
        #     for line_positions in total_positions:
        #         for single_tokens in line_positions:
        #             single_tokens += ([position_padding_index] * (max_token_lens - len(single_tokens)))
        #         line_positions += ([[position_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_positions))])

        #     total_gather_indexs = torch.LongTensor(total_gather_indexs)
        #     total_chld_prt_tokens = torch.LongTensor(total_chld_prt_tokens)
        #     total_types = torch.LongTensor(total_types)
        #     total_positions = torch.LongTensor(total_positions)

        #     # print("total_gather_indexs")
        #     # print(total_gather_indexs)
        #     # print("total_chld_prt_tokens")
        #     # print(total_chld_prt_tokens)
        #     # print("total_types")
        #     # print(total_types)
        #     # print("total_positions")
        #     # print(total_positions)
        #     # exit()
        #     samples[sample_idx]['net_input']['gather_index'] = total_gather_indexs
        #     samples[sample_idx]['net_input']['chld_prt_tokens'] = total_chld_prt_tokens
        #     samples[sample_idx]['net_input']['types'] = total_types
        #     samples[sample_idx]['net_input']['positions'] = total_positions

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)
        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % args.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break
    # input_queue.queue.clear()
    for process in processes:
        input_queue.put('close')
    while(not input_queue.empty()):
        pass
    while(not output_queue.empty()):
        try:
            samples = output_queue.get()
            del samples
        except:
            break

    for process in processes:
        process.join()
        process.close()
    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    torch.cuda.empty_cache()
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    src_dict = task.src_dict
    tgt_dict = copy.deepcopy(task.tgt_dict)
    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    need_partition_rate = 0.4
    partition_rate = 0.2
    merge_rate = 0.5
    max_sub_position = 98
    pack = bpe_pack.Bpepack('/user/fairseq/examples/translation/News_orig/tmp/codes')
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        
        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                # print(sample)
                # exit()
                sentence_len = sample['net_input']['src_tokens'].size(1)
                total_gather_indexs = []
                total_chld_prt_tokens = []
                total_types = []
                total_positions = []
                max_nums = 1
                max_token_lens = 1
                max_gather_index_lens = 1
                padding_idx = 99999
                for line_idx, line_tokens in enumerate(sample['net_input']['src_tokens']):
                    # print([word_idx for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)])
                    words = [src_dict[word_idx] for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==src_dict.eos_index or word_idx==src_dict.pad_index)]
                    gather_index_helper = [i for i in range(0, sentence_len)]
                    partition_words = []
                    merge_words = []

                    #todo 拆开与合并不要互相影响
                    # Partition
                    for idx, word in enumerate(words):
                        # if gather_index_helper[idx] > sentence_len:
                        #     continue
                        if word.replace('@@', '').__len__() > 6 and random.random() < need_partition_rate: # 先确定要拆成多少个token，随机选择位置，用负数的数值指示token个数，用于后续的整理
                            num_tokens = 1
                            init_idx = 0
                            for i in range(word.replace('@@', '').__len__() - 1):
                                now_rate = partition_rate
                                if i + 1 - init_idx == 1 or word.replace('@@', '').__len__() - (i + 1) == 1:
                                    now_rate = partition_rate / 4
                                    # now_rate = 0

                                if random.random() < now_rate and (word[init_idx:i+1] + '@@') not in src_dict:
                                    num_tokens += 1
                                    partition_words.append(word[init_idx:i+1] + '@@')
                                    init_idx = i + 1
                            if init_idx != 0:
                                partition_words.append(word[init_idx:])
                                gather_index_helper[idx] = -num_tokens
                                for i in range(num_tokens):
                                    gather_index_helper.append(len(gather_index_helper))

                    # Merge
                    idx = 0
                    while idx < words.__len__():
                        if '@@' not in words[idx] or gather_index_helper[idx] < 0:
                            idx += 1
                            continue
                        now_idx = idx
                        while '@@' in words[now_idx] and random.random() < merge_rate and gather_index_helper[idx] > 0:
                            now_idx += 1
                        if now_idx == idx:
                            idx += 1
                            continue
                        gather_index_helper.append(len(gather_index_helper))
                        for i in range(idx, now_idx + 1):
                            if i == idx:
                                gather_index_helper[i] = gather_index_helper[-1]
                            else:
                                gather_index_helper[i] = '#'
                        if '@@' in words[now_idx]:
                            merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]])+'@@')
                        else:
                            merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]]))
                        idx = now_idx + 1
                    
                    # get gather index
                    gather_index = []
                    helper_index = sentence_len
                    for idx in range(sentence_len):
                        helper = gather_index_helper[idx]
                        if helper == '#':
                            continue
                        if helper >= 0:
                            gather_index.append(helper)
                        else:
                            for i in range(-helper):
                                gather_index.append(gather_index_helper[helper_index])
                                helper_index += 1
                    
                    if len(gather_index) > max_gather_index_lens:
                        max_gather_index_lens = len(gather_index)
                    total_gather_indexs.append(gather_index)

                    this_chld_prt_tokens = []
                    this_types = []
                    this_positions = []
                    now_nums = len(partition_words) + len(merge_words)
                    if now_nums > max_nums:
                        max_nums = now_nums

                    for new_word in (partition_words + merge_words):
                        chlds, chlds_idx = find_chld(new_word, src_dict, pack)
                        prts, prts_idx = find_prt(new_word, src_dict.symbols, src_dict.indices, max_len=10-len(chlds_idx))
                        # print(new_word)
                        # print(chlds)
                        # print(prts)
                        # exit()

                        now_len = len(chlds_idx) + len(prts_idx)
                        if now_len > max_token_lens:
                            max_token_lens = now_len
                        tokens = chlds_idx + prts_idx
                        if len(tokens) == 0:
                            tokens = [src_dict.pad_index]
                        types = [1] * len(chlds_idx) + [2] * len(prts_idx)
                        position = []
                        for sub in chlds:
                            if new_word.startswith(sub):
                                position.append(0)
                            elif new_word.endswith(sub.replace('##', '')):
                                position.append(1)
                            else:
                                position.append(2)
                        for prt in prts:
                            if prt.startswith(new_word):
                                position.append(3)
                            elif prt.endswith(new_word.replace('##', '')):
                                position.append(4)
                            else:
                                position.append(5)
                        # position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
                        this_chld_prt_tokens.append(tokens)
                        this_types.append(types)
                        this_positions.append(position)

                    total_chld_prt_tokens.append(this_chld_prt_tokens)
                    # total_chld_prt_tokens.append(this_chld_prt_tokens)
                    total_types.append(this_types)
                    total_positions.append(this_positions)

                # Add paddings
                token_padding_index = src_dict.pad_index
                position_padding_index = 6
                type_padding_index = 3
                for gather_index in total_gather_indexs:
                    gather_index += ([padding_idx] * (max_gather_index_lens - len(gather_index)))
                for line_chld_prt_tokens in total_chld_prt_tokens:
                    for single_tokens in line_chld_prt_tokens:
                        single_tokens += ([token_padding_index] * (max_token_lens - len(single_tokens)))
                    line_chld_prt_tokens += ([[token_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_chld_prt_tokens))])
                for line_types in total_types:
                    for single_tokens in line_types:
                        single_tokens += ([type_padding_index] * (max_token_lens - len(single_tokens)))
                    line_types += ([[type_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_types))])
                for line_positions in total_positions:
                    for single_tokens in line_positions:
                        single_tokens += ([position_padding_index] * (max_token_lens - len(single_tokens)))
                    line_positions += ([[position_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_positions))])

                total_gather_indexs = torch.LongTensor(total_gather_indexs)
                total_chld_prt_tokens = torch.LongTensor(total_chld_prt_tokens)
                total_types = torch.LongTensor(total_types)
                total_positions = torch.LongTensor(total_positions)

                sample['net_input']['gather_index'] = total_gather_indexs
                sample['net_input']['chld_prt_tokens'] = total_chld_prt_tokens
                sample['net_input']['types'] = total_types
                sample['net_input']['positions'] = total_positions

                # Decoder Input
                sentence_len = sample['net_input']['prev_output_tokens'].size(1)
                # total_gather_indexs = []
                temp_tgt_dict = copy.deepcopy(tgt_dict)
                max_num_target_tokens = 0 
                target_total_new_target_tokens = []
                target_total_chld_prt_tokens = []
                target_total_types = []
                target_total_positions = []
                max_nums = 1
                max_token_lens = 1
                # max_gather_index_lens = 1
                padding_idx = 99999
                for line_idx, line_tokens in enumerate(sample['target']):
                    # if line_idx == 0 or line_idx == 1: 
                    #     continue
                    # print([word_idx for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==task.src_dict.eos_index or word_idx==task.src_dict.pad_index)])
                    words = [tgt_dict[word_idx] for word_idx in [int(idx) for idx in line_tokens] if not (word_idx==tgt_dict.eos_index or word_idx==tgt_dict.pad_index)]
                    gather_index_helper = [i for i in range(0, sentence_len)]
                    partition_words = []
                    merge_words = []
                    new_words = []

                    #注意拆开与合并不要互相影响
                    # Partition
                    for idx, word in enumerate(words):
                        # if gather_index_helper[idx] > sentence_len:
                        #     continue

                        if word.replace('@@', '').__len__() > 2 and random.random() < need_partition_rate: # 先确定要拆成多少个token，随机选择位置，用负数的数值指示token个数，用于后续的整理
                            num_tokens = 1
                            init_idx = 0
                            for i in range(word.replace('@@', '').__len__() - 1):
                                now_rate = partition_rate
                                # if i + 1 - init_idx == 1 or word.replace('@@', '').__len__() - (i + 1) == 1:
                                #     now_rate = partition_rate / 4

                                    # now_rate = 0

                                if random.random() < now_rate and (word[init_idx:i+1] + '@@') not in tgt_dict:
                                    num_tokens += 1
                                    partition_words.append(word[init_idx:i+1] + '@@')
                                    init_idx = i + 1
                            if init_idx != 0:
                                partition_words.append(word[init_idx:])
                                gather_index_helper[idx] = -num_tokens
                                for i in range(num_tokens):
                                    gather_index_helper.append(len(gather_index_helper))

                    # Merge
                    idx = 0
                    while idx < words.__len__():
                        if '@@' not in words[idx] or gather_index_helper[idx] < 0:
                            idx += 1
                            continue
                        now_idx = idx
                        while '@@' in words[now_idx] and random.random() < merge_rate and gather_index_helper[idx] > 0:
                            now_idx += 1
                        if now_idx == idx:
                            idx += 1
                            continue
                        gather_index_helper.append(len(gather_index_helper))
                        for i in range(idx, now_idx + 1):
                            if i == idx:
                                gather_index_helper[i] = gather_index_helper[-1]
                            else:
                                gather_index_helper[i] = '#'
                        if '@@' in words[now_idx]:
                            merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]])+'@@')
                        else:
                            merge_words.append(''.join([item.replace('@@','') for item in words[idx:now_idx+1]]))
                        idx = now_idx + 1
                    
                    # get new target_tokens
                    new_target_tokens = []
                    helper_index = sentence_len
                    merge_words_idx = 0
                    partition_words_idx = 0
                    for idx in range(sentence_len):
                        helper = gather_index_helper[idx]
                        if helper == '#':
                            continue
                        if helper >= 0 and helper < sentence_len: # normal
                            if int(line_tokens[idx]) != tgt_dict.eos_index and int(line_tokens[idx]) != tgt_dict.pad_index:
                                new_target_tokens.append(int(line_tokens[idx]))
                            # gather_index.append(helper)
                        elif helper >= sentence_len: # merged
                            new_word = merge_words[merge_words_idx]
                            merge_words_idx += 1
                            if new_word not in temp_tgt_dict:
                                new_words.append(new_word)
                            new_word_index = temp_tgt_dict.add_symbol(new_word)
                            # print(new_word)
                            new_target_tokens.append(new_word_index)
                        else: # separate
                            for i in range(-helper):
                                new_word = partition_words[partition_words_idx]
                                partition_words_idx += 1
                                # print(new_word)
                                if new_word not in temp_tgt_dict:
                                    new_words.append(new_word)
                                new_word_index = temp_tgt_dict.add_symbol(new_word)
                                new_target_tokens.append(new_word_index)
                                # gather_index.append(gather_index_helper[helper_index])
                                helper_index += 1
                    if len(new_target_tokens) > max_num_target_tokens:
                        max_num_target_tokens = len(new_target_tokens)
                    # print(words)
                    # print([temp_tgt_dict[idx] for idx in new_target_tokens])
                    # exit()
                    # if len(gather_index) > max_gather_index_lens:
                    #     max_gather_index_lens = len(gather_index)
                    # total_gather_indexs.append(gather_index)

                    # print('new words length:', [partition_words + merge_words].__len__())
                    target_this_chld_prt_tokens = []
                    target_this_types = []
                    target_this_positions = []
                    now_nums = len(partition_words) + len(merge_words)
                    if now_nums > max_nums:
                        max_nums = now_nums

                    for new_word in new_words:
                        chlds, chlds_idx = find_chld(new_word, tgt_dict, pack)
                        prts, prts_idx = find_prt(new_word, tgt_dict.symbols, tgt_dict.indices, max_len=10-len(chlds_idx))
                        # print(new_word)
                        # print(chlds)
                        # print(prts)
                        # exit()

                        now_len = len(chlds_idx) + len(prts_idx)
                        if now_len > max_token_lens:
                            max_token_lens = now_len
                        tokens = chlds_idx + prts_idx
                        if len(tokens) == 0:
                            tokens = [tgt_dict.pad_index]
                        position = []
                        for sub in chlds:
                            if new_word.startswith(sub):
                                position.append(0)
                            elif new_word.endswith(sub.replace('##', '')):
                                position.append(1)
                            else:
                                position.append(2)
                        for prt in prts:
                            if prt.startswith(new_word):
                                position.append(3)
                            elif prt.endswith(new_word.replace('##', '')):
                                position.append(4)
                            else:
                                position.append(5)
                        types = [1] * len(chlds_idx) + [2] * len(prts_idx)
                        # position = [i for i in range(len(chlds_idx))] + [max_sub_position] * len(prts_idx)
                        target_this_chld_prt_tokens.append(tokens)
                        target_this_types.append(types)
                        target_this_positions.append(position)
                    # print(new_target_tokens)
                    target_total_new_target_tokens.append(new_target_tokens)
                    target_total_chld_prt_tokens.append(target_this_chld_prt_tokens)
                    # total_chld_prt_tokens.append(this_chld_prt_tokens)
                    target_total_types.append(target_this_types)
                    target_total_positions.append(target_this_positions)

                # Add paddings
                token_padding_index = tgt_dict.pad_index
                position_padding_index = 6
                # for gather_index in total_gather_indexs:
                #     gather_index += ([padding_idx] * (max_gather_index_lens - len(gather_index)))
                new_prev_output_tokens = []
                new_target = []
                # print(target_total_new_target_tokens[-1])
                for idx in range(len(target_total_new_target_tokens)):
                    # print(target_total_new_target_tokens[-1])
                    temp1 = [tgt_dict.eos_index] + target_total_new_target_tokens[idx] + [tgt_dict.pad_index] * (max_num_target_tokens - len(target_total_new_target_tokens[idx]))
                    temp2 = target_total_new_target_tokens[idx] + [tgt_dict.eos_index] + [tgt_dict.pad_index] * (max_num_target_tokens - len(target_total_new_target_tokens[idx]))
                    # print(temp1)
                    new_prev_output_tokens.append(temp1)
                    new_target.append(temp2)
                # exit()
                # print(target_total_new_target_tokens)

                t_total_chld_prt_tokens = []
                for line_chld_prt_tokens in target_total_chld_prt_tokens:
                    for single_tokens in line_chld_prt_tokens:
                        single_tokens += ([token_padding_index] * (max_token_lens - len(single_tokens)))
                    t_total_chld_prt_tokens += line_chld_prt_tokens
                    # line_chld_prt_tokens += ([[token_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_chld_prt_tokens))])
                t_total_types = []
                for line_types in target_total_types:
                    for single_tokens in line_types:
                        single_tokens += ([type_padding_index] * (max_token_lens - len(single_tokens)))
                    t_total_types += line_types
                    # line_types += ([[type_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_types))])
                t_total_positions = []
                for line_positions in target_total_positions:
                    for single_tokens in line_positions:
                        single_tokens += ([position_padding_index] * (max_token_lens - len(single_tokens)))
                    t_total_positions += line_positions
                    # line_positions += ([[position_padding_index for i in range(max_token_lens)] for j in range(max_nums - len(line_positions))])
    
                # total_gather_indexs = torch.LongTensor(total_gather_indexs)
                target_total_chld_prt_tokens = torch.LongTensor(t_total_chld_prt_tokens)
                target_total_types = torch.LongTensor(t_total_types)
                target_total_positions = torch.LongTensor(t_total_positions)
                # print(new_prev_output_tokens[-1])
                # print(target_total_new_target_tokens)
                # print(new_prev_output_tokens.__len__())
                # print([len(item) for item in new_prev_output_tokens])
                new_prev_output_tokens = torch.LongTensor(new_prev_output_tokens)
                new_target = torch.LongTensor(new_target)

                # sample['net_input']['gather_index'] = total_gather_indexs
                sample['net_input']['target_chld_prt_tokens'] = target_total_chld_prt_tokens
                sample['net_input']['target_types'] = target_total_types
                sample['net_input']['target_positions'] = target_total_positions

                sample['net_input']['prev_output_tokens'] = new_prev_output_tokens
                sample['target'] = new_target
                trainer.task.sequence_generator.update_dict(temp_tgt_dict)
                print('temp tgt dict len:', temp_tgt_dict.__len__())
                print('tgt dict len:', tgt_dict.__len__())
                print('new tokens:', int(target_total_chld_prt_tokens.size(0)))
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
        trainer.task.sequence_generator.update_dict(tgt_dict)
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats


def cli_main(modify_parser=None):
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()

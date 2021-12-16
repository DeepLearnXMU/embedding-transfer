import subword_nmt.apply_bpe
import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings
import random
import tempfile
from multiprocessing import Pool, cpu_count

class Bpepack(object):
    def __init__(self, codes_name):
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        newdir = os.path.join(currentdir, 'subword_nmt')
        if os.path.isdir(newdir):
            warnings.simplefilter('default')
            warnings.warn(
                "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
                DeprecationWarning
            )

        # # python 2/3 compatibility
        # if sys.version_info < (3, 0):
        #     sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        #     sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        #     sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
        # else:
        #     sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        #     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        #     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

        # parser = self.create_parser()
        # args = parser.parse_args()

        # if args.num_workers <= 0:
        #     args.num_workers = cpu_count()

        # read/write files as UTF-8
        codes = codecs.open(codes_name, encoding='utf-8')
        # if args.input.name != '<stdin>':
        #     args.input = codecs.open(args.input.name, encoding='utf-8')
        # if args.output.name != '<stdout>':
        #     args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
        # if args.vocabulary:
        #     args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

        # if args.vocabulary:
        #     vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
        # else:
        #     vocabulary = None

        # if sys.version_info < (3, 0):
        #     args.separator = args.separator.decode('UTF-8')
        #     if args.glossaries:
        #         args.glossaries = [g.decode('UTF-8') for g in args.glossaries]
        #     if args.num_workers > 1:
        #         args.num_workers = 1
        #         warnings.warn("Parallel mode is only supported in Python3. Using 1 processor instead.")

        # if args.seed is not None:
        #     random.seed(args.seed)

        # self.bpe = subword_nmt.apply_bpe.BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries)
        self.bpe = subword_nmt.apply_bpe.BPE(codes, -1, '@@', None, None)


    def cut_word(self, word):
        return self.bpe.process_line(word, dropout=0)

    # def create_parser(self, subparsers=None):
    #     if subparsers:
    #         parser = subparsers.add_parser('apply-bpe',
    #             formatter_class=argparse.RawDescriptionHelpFormatter,
    #             description="learn BPE-based word segmentation")
    #     else:
    #         parser = argparse.ArgumentParser(
    #             formatter_class=argparse.RawDescriptionHelpFormatter,
    #             description="learn BPE-based word segmentation")

    #     parser.add_argument(
    #         '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
    #         metavar='PATH',
    #         help="Input file (default: standard input).")
    #     parser.add_argument(
    #         '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
    #         help="File with BPE codes (created by learn_bpe.py).")
    #     parser.add_argument(
    #         '--merges', '-m', type=int, default=-1,
    #         metavar='INT',
    #         help="Use this many BPE operations (<= number of learned symbols)"+
    #             "default: Apply all the learned merge operations")
    #     parser.add_argument(
    #         '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
    #         metavar='PATH',
    #         help="Output file (default: standard output)")
    #     parser.add_argument(
    #         '--separator', '-s', type=str, default='@@', metavar='STR',
    #         help="Separator between non-final subword units (default: '%(default)s'))")
    #     parser.add_argument(
    #         '--vocabulary', type=argparse.FileType('r'), default=None,
    #         metavar="PATH",
    #         help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    #     parser.add_argument(
    #         '--vocabulary-threshold', type=int, default=None,
    #         metavar="INT",
    #         help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    #     parser.add_argument(
    #         '--dropout', type=float, default=0,
    #         metavar="P",
    #         help="Dropout BPE merge operations with probability P (Provilkov et al., 2019). Use this on training data only.")
    #     parser.add_argument(
    #         '--glossaries', type=str, nargs='+', default=None,
    #         metavar="STR",
    #         help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
    #             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
    #             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")
    #     parser.add_argument(
    #         '--seed', type=int, default=None,
    #         metavar="S",
    #         help="Random seed for the random number generators (e.g. for BPE dropout with --dropout).")
    #     parser.add_argument(
    #         '--num-workers', type=int, default=1,
    #         help="Number of processors to process texts, only supported in Python3. If -1, set `multiprocessing.cpu_count()`. (default: %(default)s)")

    #     return parser

# ================================================================================
# Copyright 2018 Alibaba Inc. All Rights Reserved.
# ================================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import List
from silkflow_framework.sdk.base_pipeline import BasePipeline
from silkflow_framework.sdk.expt_manager import ExptManager
from silkflow_framework.sdk.xopen import xopen
import silkflow_framework.sdk.param_utils as param_utils
import silkflow_framework.sdk.helper_utils as helper_utils

class PipelineTemplate(BasePipeline):
    """A pipeline template to run train"""

    def __init__(self, expt_manager: ExptManager, prefix: str, dep_ops: List = [],
                 configs: dict = {}, params: dict = {}):
        if not prefix:
            prefix = self.__class__.__name__
        default_params = {
        }
        silkflow_detail_dir = '%s/silkflow_detail' % (os.getcwd())
        expt_manager = ExptManager(expt_dir = silkflow_detail_dir)
        super()._init(expt_manager, prefix, dep_ops=dep_ops, configs=configs, params=params,
                      default_params=default_params)

    def _define(self):
        gpu_train = super()._add_op(name = "poattention_embed_gen_fairseq-20210126-16_32_11",
                                    image = "reg.docker.alibaba-inc.com/silkflow/pytorch:1.4-cuda10.1-cudnn7-devel",
                                    command = "cd /user/vocabs_experiments/poattention_embed_gen_fairseq;sh silkflow_News.sh",
                                    gpus = 1,
                                    requirements = "cython hydra-core omegaconf fastBPE sacremoses subword_nmt sacrebleu",
                                    cpu = "10",
                                    memory = "26",
                                    node_selector = {}
        )

import logging
import os
from time import time

import hydra
from hydra.utils import get_original_cwd
from network import Network, utils
from omegaconf import DictConfig, OmegaConf, open_dict
from postprocess import NMS
from preprocess import PreProcess
from tools.utils import io

log = logging.getLogger('pipeline')

def get_latest_input_cfg(prev_stage_cfg):
    input_cfg = OmegaConf.create()
    prev_stage_dir = os.path.dirname(prev_stage_cfg.path)
    folder, _ = utils.get_latest_file_with_datetime(prev_stage_dir, '', subdir=prev_stage_cfg.inference.folder_name, ext='.h5')
    input_dir = os.path.join(prev_stage_dir, folder, prev_stage_cfg.inference.folder_name)
    input_cfg.train = os.path.join(input_dir, 'train_' + prev_stage_cfg.inference.inference_result)
    input_cfg.val = os.path.join(input_dir, 'val_' + prev_stage_cfg.inference.inference_result)
    input_cfg.test = os.path.join(input_dir, 'test_' + prev_stage_cfg.inference.inference_result)
    input_cfg.prev_stage_dir = prev_stage_dir
    return input_cfg


@hydra.main(config_path='configs', config_name='pipeline', version_base='1.1')
def main(cfg: DictConfig):
    OmegaConf.update(cfg, "paths.dataset_dir", io.to_abs_path(cfg.paths.dataset_dir, get_original_cwd()))
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))

    utils.set_random_seed(cfg.network.random_seed)

    if cfg.preprocess.run:
        assert io.folder_exist(cfg.paths.preprocess.input_dir), "Dataset directory doesn't exist"
        io.ensure_dir_exists(cfg.paths.preprocess.output_dir)

        start = time()
        preprocess_stage1 = PreProcess(cfg.preprocess, cfg.paths.preprocess)
        preprocess_stage1.process(cfg.dataset.name)
        end = time()

        duration_time = utils.duration_in_hours(end - start)
        log.info(f'Preprocess: time duration {duration_time}')

    if cfg.network.stage1.run:
        stage1_network = Network(cfg.network.stage1, cfg.paths.preprocess.output)
        if not cfg.network.stage1.eval_only:
            stage1_network.train()
        stage1_network.inference()

    if cfg.network.stage2.run:
        stage2_input_cfg = get_latest_input_cfg(cfg.paths.network.stage1)
        """
        stage2_input_cfg.train = f"path/to/current_inference/stage1/inference/train_inference_result.h5"
        stage2_input_cfg.val = f"path/to/current_inference/stage1/inference/val_inference_result.h5"
        stage2_input_cfg.test = f"path/to/current_inference/stage1/inference/val_inference_result.h5"
        stage2_input_cfg.prev_stage_dir = f"path/to/current_inference/stage1"
        """

        stage2_network = Network(cfg.network.stage2, stage2_input_cfg)
        if not cfg.network.stage2.eval_only:
            stage2_network.train()
        stage2_network.inference()

    if cfg.network.stage3.run:
        stage3_input_cfg = get_latest_input_cfg(cfg.paths.network.stage2)
        """stage3_input_cfg.train = f"path/to/current_inference/stage2/inference/train_inference_result.h5"
        stage3_input_cfg.val = f"path/to/current_inference/stage2/inference/val_inference_result.h5"
        stage3_input_cfg.test = f"path/to/current_inference/stage2/inference/val_inference_result.h5"
        stage3_input_cfg.prev_stage_dir = f"path/to/shape2motion-pytorch/current_inference/stage2""""

        stage3_network = Network(cfg.network.stage3, stage3_input_cfg)
        if not cfg.network.stage3.eval_only:
            stage3_network.train()
        stage3_network.inference()
    
    """Inference"""
    """input_cfg = OmegaConf.create()
    prev_stage_dir = os.path.dirname("path/to/network/stage2/date")
    folder, _ = utils.get_latest_file_with_datetime(prev_stage_dir, '', subdir="inference", ext='.h5')
    input_dir = os.path.join(prev_stage_dir, folder, "inference")
    input_cfg.train = os.path.join(input_dir, 'train_' + "inference_result.h5")
    input_cfg.val = os.path.join(input_dir, 'val_' + "inference_result.h5")
    input_cfg.prev_stage_dir = prev_stage_dir
    stage3_network = Network(cfg.network.stage3, stage3_input_cfg)"""

    if cfg.postprocess.run:
        log.info(f'Postprocess start')
        start = time()
        stage1_output_cfg = get_latest_input_cfg(cfg.paths.network.stage1)

        os.makedirs(f"path/to/current_inference/stage1/inference", exist_ok=True)
        stage1_output_cfg.train = f"path/to/current_inference/stage1/inference/train_inference_result.h5"
        stage1_output_cfg.val = f"path/to/current_inference/stage1/inference/val_inference_result.h5"
        stage1_output_cfg.test = f"path/to/current_inference/stage1/inference/val_inference_result.h5"
        stage1_output_cfg.prev_stage_dir = f"path/to/current_inference/stage1"

        os.makedirs(f"path/to/current_inference/stage2/inference", exist_ok=True)
        stage2_output_cfg = get_latest_input_cfg(cfg.paths.network.stage2)
        stage2_output_cfg.train = f"path/to/current_inference/stage2/inference/train_inference_result.h5"
        stage2_output_cfg.val = f"path/to/current_inference/stage2/inference/val_inference_result.h5"
        stage2_output_cfg.test = f"path/to/current_inference/stage2/inference/val_inference_result.h5"
        stage2_output_cfg.prev_stage_dir = f"path/to/current_inference/stage2"
        
        os.makedirs(f"path/to/current_inference/stage3/inference", exist_ok=True)
        stage3_output_cfg = get_latest_input_cfg(cfg.paths.network.stage3)
        stage3_output_cfg = get_latest_input_cfg(cfg.paths.network.stage2)
        stage3_output_cfg.train = f"path/to/current_inference/stage3/inference/train_inference_result.h5"
        stage3_output_cfg.val = f"path/to/current_inference/stage3/inference/val_inference_result.h5"
        stage3_output_cfg.test = f"path/to/current_inference/stage3/inference/val_inference_result.h5"
        stage3_output_cfg.prev_stage_dir = f"path/to/current_inference/stage3"

        nms_cfg = cfg.postprocess.nms

        with open_dict(nms_cfg):
            nms_cfg.stage1 = stage1_output_cfg
            nms_cfg.stage2 = stage2_output_cfg
            nms_cfg.stage3 = stage3_output_cfg
        io.ensure_dir_exists(cfg.paths.postprocess.path)
        nms = NMS(nms_cfg)
        # data_sets = ['train', cfg.network.test_split]
        data_sets = [cfg.network.test_split]
        for data_set in data_sets:
            output_path = os.path.join(cfg.paths.postprocess.path, f'{data_set}_' + cfg.paths.postprocess.output.nms_result)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_path = f"path/to/current_inference/nms/inference/{data_set}_nms_result.h5"
            nms.process(output_path, data_set)
        end = time()

        duration_time = utils.duration_in_hours(end - start)
        log.info(f'Postprocess: time duration {duration_time}')


if __name__ == '__main__':
    start = time()
    main()
    end = time()

    duration_time = utils.duration_in_hours(end - start)
    log.info(f'Pipeline: Total time duration {duration_time}')

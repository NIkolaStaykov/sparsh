# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

import wandb
from lightning.fabric import seed_everything

from tactile_ssl.utils import get_local_rank

from tactile_ssl.utils.logging import get_pylogger, print_config_tree
from tactile_ssl.trainer import Trainer

logger = get_pylogger(__name__)


def init_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        dir=cfg.save_dir,
        id=f"{cfg.id}_{get_local_rank()}",
        group=cfg.group,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    return wandb


def get_dataloader_reskin(cfg: DictConfig):
    data_cfg = cfg.data

    train_dataset_list = data_cfg.train_dataset_list
    val_dataset_list = data_cfg.val_dataset_list
    train_datasets, val_datasets = [], []
    for dataset_name in train_dataset_list:
        data_path = os.path.join(data_cfg.dataset.data_path, dataset_name)
        train_datasets.append(
            hydra.utils.instantiate(data_cfg.dataset, data_path=data_path)
        )
    for dataset_name in val_dataset_list:
        data_path = os.path.join(data_cfg.dataset.data_path, dataset_name)
        val_datasets.append(
            hydra.utils.instantiate(data_cfg.dataset, data_path=data_path)
        )
    train_dset = data.ConcatDataset(train_datasets)
    val_dset = data.ConcatDataset(val_datasets)
    return train_dset, val_dset


def split_trajectories(dset, test_frac, seed):
    """Split a force/slip dataset by whole trajectories. Returns (train_idx, test_idx, test_trajectories)."""
    trajs = sorted(dset.trajectories.keys())
    rng = np.random.default_rng(seed)
    test_trajs = set(rng.choice(trajs, int(round(test_frac * len(trajs))), replace=False).tolist())
    is_test = np.array([dset.idx2traj[i]["trajectory"] in test_trajs for i in range(len(dset))])
    return np.where(~is_test)[0], np.where(is_test)[0], sorted(test_trajs)


def get_dataloader_digit(cfg: DictConfig):
    data_cfg = cfg.data
    # optional trajectory-level split of one dataset: {dataset, test_frac, seed}
    traj_split = data_cfg.get("traj_split", None)
    datasets = []
    list_datasets = data_cfg.dataset.config.list_datasets
    path_dataset = data_cfg.dataset.config.path_dataset

    
    for d in list_datasets:
        if data_cfg.dataset.config.look_in_folder:
            obj = d
            logger.info(f"Loading train dataset: {obj}")
            files_list = os.listdir(os.path.join(path_dataset, obj))
            
            for f in files_list:
                dataset_name = obj + "/" + f.split(".")[0]
                try:
                    datasets.append(
                        hydra.utils.instantiate(
                            data_cfg.dataset, dataset_name=dataset_name
                        )
                    )
                except Exception as e:
                    logger.error(f"Error loading {dataset_name}: {e}")

        else:
            dset = hydra.utils.instantiate(data_cfg.dataset, dataset_name=d)
            if traj_split is not None and d == traj_split.dataset:
                train_idx, _, _ = split_trajectories(dset, traj_split.test_frac, traj_split.seed)
                logger.info(f"Trajectory split of {d}: {len(train_idx)} of {len(dset)} samples kept for training")
                dset = data.Subset(dset, train_idx)
            datasets.append(dset)
        
    train_dset_samples = sum([len(d) for d in datasets])

    # if dataset is larger than maximum budget of training data, sample uniformly
    data_cfg.max_train_data = train_dset_samples if data_cfg.max_train_data == None else data_cfg.max_train_data
    if train_dset_samples > data_cfg.max_train_data:
        try:
            idxs = [np.random.choice(len(d), data_cfg.max_train_data//len(datasets), replace=False) for d in datasets]
            datasets = [data.Subset(d, idxs[i]) for i, d in enumerate(datasets)]
            dataset = data.ConcatDataset(datasets)
        except:
            dataset = data.ConcatDataset(datasets)
            dataset, _ = data.random_split(
                dataset, [data_cfg.max_train_data, len(dataset) - data_cfg.max_train_data]
            )
    else:
        dataset = data.ConcatDataset(datasets)
    logger.info(f"Loaded dataset size: {len(dataset)}")

    dataset_test = None
    list_datasets_test = data_cfg.dataset.config.list_datasets_test
    if len(list_datasets_test) > 0:
        dataset_test = []
        for d in list_datasets_test:
            if data_cfg.dataset.config.look_in_folder:
                obj = d
                logger.info(f"Loading test dataset: {obj}")
                files_list = os.listdir(os.path.join(path_dataset, obj))
            
                for f in files_list:
                    dataset_name = obj + "/" + f.split(".")[0]
                    try:
                        dataset_test.append(
                            hydra.utils.instantiate(
                                data_cfg.dataset, dataset_name=dataset_name
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error loading {dataset_name}: {e}")
            else:
                dataset_test.append(
                    hydra.utils.instantiate(data_cfg.dataset, dataset_name=d)
                )
        dataset_test = data.ConcatDataset(dataset_test)

        test_idx = np.arange(len(dataset_test))
        if traj_split is not None and list(list_datasets_test) == [traj_split.dataset]:
            train_idx, test_idx, test_trajs = split_trajectories(
                dataset_test.datasets[0], traj_split.test_frac, traj_split.seed
            )
            all_trajs = sorted(dataset_test.datasets[0].trajectories.keys())
            with open(os.path.join(cfg.paths.output_dir, "traj_split.json"), "w") as f:
                json.dump({"dataset": traj_split.dataset, "test": [int(t) for t in test_trajs],
                           "train": [int(t) for t in all_trajs if t not in set(test_trajs)]}, f)
            logger.info(f"Trajectory split of {traj_split.dataset}: testing on {len(test_trajs)} held-out trajectories ({len(test_idx)} samples)")
        # optional fixed-size test subset (own seeded RNG, so indices are reproducible outside training)
        max_test_data = data_cfg.get("max_test_data", None)
        if max_test_data is not None and len(test_idx) > max_test_data:
            rng = np.random.default_rng(data_cfg.get("test_subset_seed", 0))
            test_idx = np.sort(rng.choice(test_idx, max_test_data, replace=False))
        if len(test_idx) < len(dataset_test):
            np.save(os.path.join(cfg.paths.output_dir, "test_subset_idxs.npy"), test_idx)
            dataset_test = data.Subset(dataset_test, test_idx)

    if dataset_test is None:
        train_dset_size = int(len(dataset) * data_cfg.train_val_split)
        train_dset, val_dset = data.random_split(
            dataset, [train_dset_size, len(dataset) - train_dset_size]
        )
    else:
        train_dset = dataset
        val_dset = dataset_test


    # adjust training dataset size
    train_dset_size = int(len(train_dset) * data_cfg.train_data_budget)
    train_dset, _ = data.random_split(
        train_dset, [train_dset_size, len(train_dset) - train_dset_size]
    )

    try:
        val_dset_size = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(
            val_dset, [val_dset_size, len(val_dset) - val_dset_size]
        )
    except:
        pass

    logger.info("Train/Val data split")
    logger.info(f"Train dataset size: {len(train_dset)}")
    logger.info(f"Val dataset size: {len(val_dset)}")

    return train_dset, val_dset


def get_dataloaders(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor == "digit" or "gelsight" in data_cfg.sensor :
        train_dset, val_dset = get_dataloader_digit(cfg)
    elif data_cfg.sensor == "reskin":
        train_dset, val_dset = get_dataloader_reskin(cfg)
    else:
        raise NotImplementedError("Sensor type not implemented yet.")

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)

    return train_dataloader, val_dataloader


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(
                f"Unable to resume: No checkpoints found for experiment with id {job_id}"
            )
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/wandb/"):
            logger.warning(
                f"Unable to resume: No wandb logs found for experiment with id {job_id}"
            )
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning(
                "Could not find a config.yaml file in the resume directory. Using the current config."
            )
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")

        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        experiment_name = cfg.experiment_name
        cfg.wandb.id = f"{job_id}_{experiment_name}"
        logger.info(
            f"Resuming experiment {job_id} with wandb_id: {cfg.wandb.id} from latest checkpoint at {cfg.ckpt_path}"
        )
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating wandb ...")
    wandb = init_wandb(cfg.wandb)
    if not resume_state:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logger.info(
        f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>"
    )
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    wandb.finish()


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

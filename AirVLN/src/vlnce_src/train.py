import os
import sys
from pathlib import Path
sys.path.append(str(Path(str(os.getcwd())).resolve()))
import gc
import time
import lmdb
import tqdm
import math
import random
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from typing import List, Optional, DefaultDict
import msgpack_numpy

from utils.logger import logger
from utils.utils import get_rank, is_dist_avail_and_initialized, is_main_process, init_distributed_mode
from Model.il_trainer import VLNCETrainer
from Model.utils.tensor_dict import DictTree, TensorDict
from Model.aux_losses import AuxLosses
from Model.utils.tensorboard_utils import TensorboardWriter
from Model.utils.common import observations_to_image, append_text_to_image, generate_video

from src.common.param import args
from src.vlnce_src.env import AirVLNENV
from src.vlnce_src.util import read_vocab, Tokenizer


def setup():
    init_distributed_mode()

    seed = 100 + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False


class DDPIWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
        ignore_episode_ids = []
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw:
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])

        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                if key.decode() in ignore_episode_ids:
                    continue
                else:
                    self.keys.append(key.decode())

        self.length = len(self.keys)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.start = 0
        self.end = self.length

        self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        self.iter_start = 0 + self.rank * self.per_worker
        self.iter_end = min(self.iter_start + self.per_worker, self.end)
        logger.warning("END init DDP-Dataset \t rank: {} \t start({}) - end({})".format(self.rank, self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0:
                        logger.warning("rank: {} \t lmdb load: {} / {}".format(self.rank, i+1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    )

                    lengths.append(len(new_preload[-1][0]))

            sort_priority = list(range(len(lengths)))
            random.shuffle(sort_priority)

            sorted_ordering = list(range(len(lengths))) # sort by length to avoid to much pad? weired
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop() 

    def __next__(self):
        obs, prev_actions, oracle_actions = self._load_next()

        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        # first action is important inflection is important
        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        )

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections],
        )

    def __iter__(self):
        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(self.iter_start, self.iter_end)), self.preload_size)
            )
        )

        return self


class IWTrajectoryDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        lmdb_features_dir,
        use_iw=True,
        inflection_weight_coef=1.0,
        lmdb_map_size=5.0e12,
        batch_size=1,
        ignore_episode_ids = []
    ):
        super().__init__()

        self.lmdb_features_dir = lmdb_features_dir
        self.lmdb_map_size = lmdb_map_size
        self.preload_size = batch_size * 100  # preload size
        self._preload = []
        self.batch_size = batch_size

        self.keys = []
        self.seed = 1

        if use_iw: # use inflection weight for importance weighting
            self.inflec_weights = torch.tensor([1.0, inflection_weight_coef])
        else:
            self.inflec_weights = torch.tensor([1.0, 1.0])
        # read keys from lmdb
        with lmdb.open(
            self.lmdb_features_dir,
            map_size=int(self.lmdb_map_size),
            readonly=True,
            lock=False,
            readahead=False,
        ) as lmdb_env, tqdm.tqdm(
            total=int(lmdb_env.stat()["entries"]), dynamic_ncols=True
        ) as pbar, lmdb_env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                pbar.update()
                if key.decode() in ignore_episode_ids:
                    continue
                else:
                    self.keys.append(key.decode()) # only keys
        print('ignore_episode_ids',ignore_episode_ids)
        self.length = len(self.keys) # keys = trajectory_id airvln-s 10113

        # import ipdb; ipdb.set_trace()
        self.iter_start = 0
        self.iter_end = self.length
        logger.warning("END init Dataset \t start({}) - end({})".format(self.iter_start, self.iter_end))

    def _load_next(self):
        if len(self._preload) == 0:
            if len(self.load_ordering) == 0:
                raise StopIteration

            new_preload = []
            lengths = []
            with lmdb.open(
                self.lmdb_features_dir,
                map_size=int(self.lmdb_map_size),
                readonly=True,
                lock=False,
            ) as lmdb_env, lmdb_env.begin(buffers=True) as txn:
                for i in range(self.preload_size):
                    if len(self.load_ordering) == 0:
                        break

                    if (i+1) % 10 == 0: # log per 10 bathc
                        if self.worker_info is not None:
                            logger.info("{} lmdb load: {} / {}".format(self.worker_info.id, i+1, self.preload_size))
                        else:
                            logger.info("{} lmdb load: {} / {}".format(0, i+1, self.preload_size))

                    new_preload.append(
                        msgpack_numpy.unpackb(
                            txn.get(str(self.keys[self.load_ordering.pop()]).encode()),
                            raw=False,
                        )
                    ) # get new key-value pair from lmdb to new_preload  list obs, prev_actions, oracle_actions

                    lengths.append(len(new_preload[-1][0]))
            # import ipdb ; ipdb.set_trace()
            sort_priority = list(range(len(lengths))) # self.preload_size
            random.shuffle(sort_priority) # priority bwteen same lengths

            sorted_ordering = list(range(len(lengths)))
            sorted_ordering.sort(key=lambda k: (lengths[k], sort_priority[k]))

            for idx in _block_shuffle(sorted_ordering, self.batch_size):
                self._preload.append(new_preload[idx])

            del new_preload, lengths

        return self._preload.pop() # trajecotry of same length in one batch

    def __next__(self): # return one sample
        
        obs, prev_actions, oracle_actions = self._load_next() # whole trajectory
        # import ipdb; ipdb.set_trace()
        for k, v in obs.items():
            obs[k] = torch.from_numpy(np.copy(v))

        prev_actions = torch.from_numpy(np.copy(prev_actions))
        oracle_actions = torch.from_numpy(np.copy(oracle_actions))

        inflections = torch.cat(
            [
                torch.tensor([1], dtype=torch.long),
                (oracle_actions[1:] != oracle_actions[:-1]).long(),
            ]
        ) # is the point is inflection, choose weight

        return (
            obs,
            prev_actions,
            oracle_actions,
            self.inflec_weights[inflections], # choose inflection weights
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.worker_info = worker_info
        if worker_info is None: # one process/main process
            start = 0
            end = self.length
        else: # multiple processes worker 1: start=250, end=500
            per_worker = int(np.ceil(self.length / worker_info.num_workers))

            start = per_worker * worker_info.id
            end = min(start + per_worker, self.length)

        # Reverse so we can use .pop()
        self.load_ordering = list(
            reversed(
                _block_shuffle(list(range(start, end)), self.preload_size)
            )
        )

        return self


class ObservationsDict(dict):
    def pin_memory(self):
        for k, v in self.items():
            self[k] = v.pin_memory()

        return self


def collate_fn(batch):
    """Each sample in batch: (
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    list of sample to batch, deal with padding
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        ) # pad with 0 to fix shape
        return torch.cat([t, pad], dim=0) # T' D -> max_len D

    transposed = list(zip(*batch)) # merge each item in batch

    observations_batch = list(transposed[0]) # B X
    prev_actions_batch = list(transposed[1])
    corrected_actions_batch = list(transposed[2])
    weights_batch = list(transposed[3])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            ) # batch_id key -- key -value dict shape = batch

    observations_batch = new_observations_batch # B D of one key-value

    # max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    # TODO define max_traj_len = 500 is neccesary if we just use action chunk
    max_traj_len = 500 # useful for airvln -s   661.8 for airvln and 321.3 airvln-s
    # padding to max trajectory length
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid][:max_traj_len, ...], max_traj_len, fill_val=1.0
            ) # T D -> max_traj_len D

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid][:max_traj_len, ...], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid][:max_traj_len, ...], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        ) # list(T D ) -> T B D
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )  # [T B D] -> [T*B D] for RNN input ?

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1) # T B
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0 # first action is useless

    observations_batch = ObservationsDict(observations_batch)

    return (
        observations_batch, # many keys T*B D
        prev_actions_batch.view(-1, 1), # T*B 1
        not_done_masks.view(-1, 1), # T*B 1
        corrected_actions_batch, # T B 
        weights_batch, # T B 
    )


def _block_shuffle(lst, block_size):
    blocks = [lst[i : i + block_size] for i in range(0, len(lst), block_size)]
    random.shuffle(blocks)

    return [ele for block in blocks for ele in block]


@torch.no_grad()
def batch_obs(
    observations: List[DictTree],
    device: Optional[torch.device] = None,
) -> TensorDict:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of torch.Tensor of observations.
    """
    batch: DefaultDict[str, List] = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(torch.as_tensor(obs[sensor]))

    batch_t: TensorDict = TensorDict()

    for sensor in batch:
        batch_t[sensor] = torch.stack(batch[sensor], dim=0)

    return batch_t.map(lambda v: v.to(device))


def initialize_tokenizer():
    if args.tokenizer_use_bert: # TODO add more tokenizers to support
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        vocab = read_vocab(args.TRAIN_VOCAB)
        tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)

    return tok


def initialize_env(split='train'):
    tok = initialize_tokenizer() # tokenizer_use_bert

    train_env = AirVLNENV(batch_size=args.batchSize, split=split, tokenizer=tok)

    return train_env


def initialize_trainer(): # define trainer model
    from gym import spaces
    from airsim_plugin.airsim_settings import AirsimActions

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(int(len(AirsimActions))) # 8

    trainer = VLNCETrainer(
        load_from_ckpt=False,
        observation_space=observation_space,
        action_space=action_space,
    ) # policy type is defined in args.policy_type, from src.common.param import args # common parameters 

    logger.info('initialize_trainer over')
    return trainer


def collect_data(data_it=0):
    logger.info(args)

    train_env = initialize_env(split='train') # define tokenizer
    trainer = initialize_trainer()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

    # pre-extract rgb and depth features 
    def hook_builder(tgt_tensor): # weird
        def hook(m, i, o):
            tgt_tensor.set_(o.cpu()) # capture the output tensor tgt_tensor=o.cpu()

        return hook

    rgb_features = torch.zeros((1,), device="cpu")
    if not args.ablate_rgb:
        # extract rgb features function? 
        # deploy/trigger hook when forward pass trainer.policy.net.rgb_encoder.layer_extract layer
        rgb_hook = trainer.policy.net.rgb_encoder.layer_extract.register_forward_hook(
            hook_builder(rgb_features)
        )
    else:
        rgb_hook = None

    depth_features = torch.zeros((1,), device="cpu")
    if not args.ablate_depth:
        depth_hook = trainer.policy.net.depth_encoder.visual_encoder.register_forward_hook(
            hook_builder(depth_features)
        )
    else:
        depth_hook = None

    p = 1.0
    beta = 1.0


    #
    with torch.no_grad():
        end_iter = len(train_env.data) # the number of data
        pbar = None
        pbar_pre_index = 0
        while train_env.index_data < end_iter: # one loop
            if pbar_pre_index + train_env.batch_size >= end_iter:
                break

            pbar_pre_index = train_env.index_data
            train_env.next_minibatch() # get banch
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if pbar is None:
                pbar = tqdm.tqdm(total=end_iter)
                pbar.update(train_env.index_data)
            else:
                pbar.update(n=train_env.index_data-pbar_pre_index)

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)] # N env
            envs_to_pause = []
            # reset envs
            outputs = train_env.reset() # reset envs output obs
            # obs: observations, infos, dones
            # what is observations key?
            # import ipdb; ipdb.set_trace()
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device) # tensor dict key? no position?

            ended = False

            for t in range(int(args.maxAction) + 1):
                logger.info('{} - {} / {}'.format(int(train_env.index_data)-int(train_env.batch_size), t, end_iter))

                for i in range(train_env.batch_size):
                    if dones[i] and not skips[i]:
                        if args.collect_type in ['TF']:
                            _episodes = episodes[i].copy() # list
                            for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                                for __i, __j in enumerate(_episodes):
                                    _episodes[__i][0]['instruction'] = _j

                                ep = _episodes.copy()
                                traj_obs = batch_obs(
                                    [step[0] for step in ep],
                                    device=torch.device("cpu"),
                                )
                                del traj_obs['teacher_action']
                                for k, v in traj_obs.items():
                                    traj_obs[k] = v.numpy()
                                
                                # for what? how get the features?
                                transposed_ep = [
                                    traj_obs,
                                    np.array([step[1] for step in ep], dtype=np.int64),
                                    np.array([step[2] for step in ep], dtype=np.int64),
                                ]

                                train_env.threading_lock_lmdb_features_txn.acquire()
                                lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                                train_env.lmdb_features_txn.put(
                                    lmdb_key.encode(),
                                    msgpack_numpy.packb(
                                        transposed_ep, use_bin_type=True
                                    ),
                                ) # why? or what 
                                train_env.lmdb_features_txn.commit()
                                train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                                train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                                train_env.threading_lock_lmdb_features_txn.release()
                                logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            _episodes = []
                            envs_to_pause.append(i)
                            skips[i] = True

                        else:
                            ep = episodes[i]
                            traj_obs = batch_obs(
                                [step[0] for step in ep],
                                device=torch.device("cpu"),
                            )
                            del traj_obs['teacher_action']
                            for k, v in traj_obs.items():
                                traj_obs[k] = v.numpy()

                            transposed_ep = [
                                traj_obs,
                                np.array([step[1] for step in ep], dtype=np.int64),
                                np.array([step[2] for step in ep], dtype=np.int64),
                            ]

                            train_env.threading_lock_lmdb_features_txn.acquire()
                            lmdb_key = str(infos[i]['episode_id'])
                            train_env.lmdb_features_txn.put(
                                lmdb_key.encode(),
                                msgpack_numpy.packb(
                                    transposed_ep, use_bin_type=True
                                ),
                            )
                            train_env.lmdb_features_txn.commit()
                            train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                            train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                            train_env.lmdb_collected_keys.add(lmdb_key)
                            train_env.threading_lock_lmdb_features_txn.release()
                            logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                            if args.run_type in ['collect'] and args.collect_type in ['TF']:
                                train_env.threading_lock_lmdb_rgb_txn.acquire()
                                train_env.lmdb_rgb_txn.commit()
                                train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                                train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                                train_env.threading_lock_lmdb_rgb_txn.release()

                                train_env.threading_lock_lmdb_depth_txn.acquire()
                                train_env.lmdb_depth_txn.commit()
                                train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                                train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                                train_env.threading_lock_lmdb_depth_txn.release()

                            episodes[i] = []
                            envs_to_pause.append(i)
                            skips[i] = True

                    if np.array(dones).all():
                        ended = True

                if ended:
                    break
                
                # get action from policy not used in collect data
                # import ipdb; ipdb.set_trace()
                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                actions = torch.where(
                    torch.rand_like(actions, dtype=torch.float) < beta,
                    batch['teacher_action'].long(), # Batchsize 1?
                    actions,
                ) # randomly select teacher action with prob beta probs = 1 choose teacher action
                # TODO add more keys in image features to train model
                # add rgb_features and depth_features to episodes observations, delte raw rgb and depth
                for i in range(train_env.batch_size):
                    if not args.ablate_rgb and rgb_features is not None:
                        observations[i]["rgb_features"] = rgb_features[i]
                        del observations[i]["rgb"] # no on 

                    if not args.ablate_depth and depth_features is not None:
                        observations[i]["depth_features"] = depth_features[i]
                        del observations[i]["depth"]

                    if i in envs_to_pause:
                        continue

                    episodes[i].append(
                        (
                            observations[i], # current observation dict
                            prev_actions[i].item(), # action from previous step
                            batch['teacher_action'][i].item(), # GT teacher action
                        )
                    ) # observations, prev_actions, teacher_action

                prev_actions.copy_(actions) # update prev_actions with actions

                # Make action and get the new state
                actions = [temp[0] for temp in actions.cpu().numpy()] # list
                train_env.makeActions(actions) # make actions in envs 

                outputs = train_env.get_obs()
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                ) # advoid done envs in next step
            # episodes batch_size * Max_action
            for i in range(train_env.batch_size): # same as for t in range(int(args.maxAction) + 1) 
                if dones[i] and not t >= int(args.maxAction):
                    continue
                # import ipdb; ipdb.set_trace()
                if args.collect_type in ['TF']:
                    _episodes = episodes[i].copy() # max_action
                    # one episode may have multiple instructions
                    for _i, _j in enumerate(train_env.trajectory_id_2_instruction_tokens[infos[i]['trajectory_id']]):
                        for __i, __j in enumerate(_episodes):
                            # _episodes[__i][0] = observations at __i timestep
                            _episodes[__i][0]['instruction'] = _j # token ids  
                        # 1 episode + multiple instructions = multiple episodes
                        ep = _episodes.copy()
                        if len(ep) <= 0:
                            continue
                        traj_obs = batch_obs(
                            [step[0] for step in ep],
                            device=torch.device("cpu"),
                        ) #observations dict to tensor dict 
                        del traj_obs['teacher_action']
                        for k, v in traj_obs.items():
                            traj_obs[k] = v.numpy()

                        transposed_ep = [
                            traj_obs, # batch observations
                            np.array([step[1] for step in ep], dtype=np.int64), # prev_actions
                            np.array([step[2] for step in ep], dtype=np.int64), # teacher_action
                        ]
                        # Lightning Memory-Mapped Database
                        train_env.threading_lock_lmdb_features_txn.acquire()
                        lmdb_key = str(train_env.trajectory_id_2_episode_ids[infos[i]['trajectory_id']][_i])
                        train_env.lmdb_features_txn.put(
                            lmdb_key.encode(), # encode to bytes
                            msgpack_numpy.packb(
                                transposed_ep, use_bin_type=True
                            ), # include observations, prev_actions, teacher_action
                        )
                        train_env.lmdb_features_txn.commit() # commit the transaction
                        # update the start id of lmdb
                        train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                        train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                        # add new key to collected keys
                        train_env.lmdb_collected_keys.add(lmdb_key)
                        train_env.threading_lock_lmdb_features_txn.release()
                        logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))
                    # add rgb and depth in env.py and maintain database in train.py
                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    _episodes = []
                    envs_to_pause.append(i)
                    skips[i] = True

                else:
                    ep = episodes[i]
                    if len(ep) <= 0:
                        continue
                    traj_obs = batch_obs(
                        [step[0] for step in ep],
                        device=torch.device("cpu"),
                    )
                    del traj_obs['teacher_action']
                    for k, v in traj_obs.items():
                        traj_obs[k] = v.numpy()

                    transposed_ep = [
                        traj_obs,
                        np.array([step[1] for step in ep], dtype=np.int64),
                        np.array([step[2] for step in ep], dtype=np.int64),
                    ]

                    train_env.threading_lock_lmdb_features_txn.acquire()
                    lmdb_key = str(infos[i]['episode_id'])
                    train_env.lmdb_features_txn.put(
                        lmdb_key.encode(),
                        msgpack_numpy.packb(
                            transposed_ep, use_bin_type=True
                        ),
                    )
                    train_env.lmdb_features_txn.commit()
                    train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                    train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                    train_env.lmdb_collected_keys.add(lmdb_key)
                    train_env.threading_lock_lmdb_features_txn.release()
                    logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                    if args.run_type in ['collect'] and args.collect_type in ['TF']:
                        train_env.threading_lock_lmdb_rgb_txn.acquire()
                        train_env.lmdb_rgb_txn.commit()
                        train_env.lmdb_rgb_start_id = train_env.lmdb_rgb_env.stat()["entries"]
                        train_env.lmdb_rgb_txn = train_env.lmdb_rgb_env.begin(write=True)
                        train_env.threading_lock_lmdb_rgb_txn.release()

                        train_env.threading_lock_lmdb_depth_txn.acquire()
                        train_env.lmdb_depth_txn.commit()
                        train_env.lmdb_depth_start_id = train_env.lmdb_depth_env.stat()["entries"]
                        train_env.lmdb_depth_txn = train_env.lmdb_depth_env.begin(write=True)
                        train_env.threading_lock_lmdb_depth_txn.release()

                    episodes[i] = []
                    envs_to_pause.append(i)
                    skips[i] = True

    try:
        pbar.close()
    except:
        pass

    if rgb_hook is not None:
        rgb_hook.remove()
    if depth_hook is not None:
        depth_hook.remove()

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass
    logger.info('END data_it: {}'.format(data_it))


def train_vlnce():
    logger.info(args)

    if get_rank() == 0: # main progress
        writer = SummaryWriter(
            log_dir=str(Path(args.project_prefix) / 'DATA/output/{}/train/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        )
    else:
        writer = None
    # Step 1: define policy and optimizer
    trainer = initialize_trainer() 
    # import ipdb; ipdb.set_trace()
    for dagger_it in range(int(args.dagger_it)):
        step_id = 0

        if torch.cuda.is_available():
            with torch.cuda.device(trainer.device):
                torch.cuda.empty_cache()
        gc.collect()
        # Step 2: make dataloader
        if args.policy_type == 'seq2seq': # need rgb extracted feature
            lmdb_features_dir = str(Path(args.project_prefix) / 'DATA/img_features_v0/collect/{}/train'.format(args.name))
        elif args.policy_type == 'cma': # need raw rgb
            lmdb_features_dir = str(Path(args.project_prefix) / 'DATA/img_features/collect/{}/train'.format(args.name))
        assert os.path.exists(str(lmdb_features_dir))
        ignore_episode_ids_path = str(Path(args.project_prefix) / 'DATA/data/aerialvln-s/train_ignore_episode_id.json')
        ignore_episode_ids = json.load(open(ignore_episode_ids_path, 'r')) if os.path.exists(ignore_episode_ids_path) else []
        if args.DistributedDataParallel:
            dataset = DDPIWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef),
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
                ignore_episode_ids=ignore_episode_ids, # ignore episode ids
            )
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )
        else:
            # import ipdb; ipdb.set_trace()
            dataset = IWTrajectoryDataset(
                lmdb_features_dir,
                use_iw=True,
                inflection_weight_coef=float(args.inflection_weight_coef), # 1.9
                lmdb_map_size=5.0e12,
                batch_size=args.batchSize,
                ignore_episode_ids=ignore_episode_ids if args.ignore_specific_episode else [], # ignore episode ids
            )
            diter = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=False,
                collate_fn=collate_fn, # padding + data processing
                pin_memory=False,
                drop_last=True,
                num_workers=0,
            )
        # import ipdb; ipdb.set_trace()
        AuxLosses.activate()
        for epoch in tqdm.trange(int(args.epochs), dynamic_ncols=True):
            batch_cnt = 0
            for batch in tqdm.tqdm(
                diter,
                total=dataset.length // dataset.batch_size if not args.DistributedDataParallel else (dataset.iter_end - dataset.iter_start) // dataset.batch_size,
                leave=False,
                dynamic_ncols=True,
            ):
                (
                    observations_batch, # ['instruction', 'progress', 'pose', 'rgb_features', 'depth_features']
                    prev_actions_batch, # 500*bs 1
                    not_done_masks, # 500*bs 1
                    corrected_actions_batch, # 500 bs
                    weights_batch, # 500 bs
                ) = batch
                # 'instruction' 500*bs  300,  500 is defacut episode length
                # 'progress' 500*bs
                # 'pose' 500*bs 7
                # 'rgb_features' 500*bs 2048 1 1
                # 'depth_features' 500*bs 128 4 4
                observations_batch = {
                    k: v.to(
                        device=trainer.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )
                    for k, v in observations_batch.items()
                }
                # Step 3: update agent
                loss, action_loss, aux_loss = trainer._update_agent(
                    observations_batch,
                    prev_actions_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                    not_done_masks.to(
                        device=trainer.device, non_blocking=True
                    ),
                    corrected_actions_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                    weights_batch.to(
                        device=trainer.device, non_blocking=True
                    ),
                )

                logger.warning(
                    'dagger_it: {} / {} \t epoch: {} / {} \t batch: {} / {}'.format(
                        dagger_it, args.dagger_it,
                        epoch, args.epochs,
                        batch_cnt, dataset.length // dataset.batch_size
                    )
                )

                logger.info(f"train_loss: {loss}")
                logger.info(f"train_action_loss: {action_loss}")
                logger.info(f"train_aux_loss: {aux_loss}")
                logger.info(f"Batches processed: {step_id}.")
                logger.info(
                    f"On DAgger iter {dagger_it}, Epoch {epoch}."
                )
                logger.info('\n')

                if get_rank() == 0:
                    writer.add_scalar(
                        f"train_loss_iter_{dagger_it}", loss, step_id
                    )
                    writer.add_scalar(
                        f"train_action_loss_iter_{dagger_it}",
                        action_loss,
                        step_id,
                    )
                    writer.add_scalar(
                        f"train_aux_loss_iter_{dagger_it}",
                        aux_loss,
                        step_id,
                    )

                step_id += 1
                batch_cnt += 1

            if is_main_process():
                if ((dagger_it * args.epochs + epoch)+1) % 5 == 0:
                    trainer.save_checkpoint(
                        f"ckpt.{dagger_it * args.epochs + epoch}.pth",
                        dagger_it,
                        epoch,
                    )

            if is_dist_avail_and_initialized() == 1:
                dist.barrier()

        if is_main_process():
            trainer.save_checkpoint(
                f"ckpt.LAST.pth",
                dagger_it,
                epoch,
            )
        AuxLosses.deactivate()


def eval_vlnce():
    logger.info(args)

    writer = TensorboardWriter(
        str(Path(args.project_prefix) / 'DATA/output/{}/eval/TensorBoard/{}'.format(args.name, args.make_dir_time)),
        flush_secs=30,
    )

    tok = initialize_tokenizer()

    assert os.path.exists(args.EVAL_CKPT_PATH_DIR), 'The eval file/folder does not exist'
    if os.path.isfile(args.EVAL_CKPT_PATH_DIR):
        from Model.utils.common import get_checkpoint_id

        # evaluate singe checkpoint
        proposed_index = get_checkpoint_id(args.EVAL_CKPT_PATH_DIR)
        # weired? 
        if proposed_index is not None:
            ckpt_idx = proposed_index
        else:
            ckpt_idx = 100000

        _eval_checkpoint(
            checkpoint_path=args.EVAL_CKPT_PATH_DIR,
            writer=writer,
            tok=tok,
            checkpoint_index=ckpt_idx,
        )
        logger.info("END evaluate")
    else: # evaluate multiple checkpoints
        from Model.utils.common import poll_checkpoint_folder

        # evaluate multiple checkpoints in order
        prev_ckpt_ind = -1
        while True:
            current_ckpt = None
            while current_ckpt is None:
                current_ckpt = poll_checkpoint_folder(
                    args.EVAL_CKPT_PATH_DIR, prev_ckpt_ind
                )
                time.sleep(2)
            logger.info(f"=======current_ckpt: {current_ckpt}=======")
            prev_ckpt_ind += 1

            if prev_ckpt_ind <= 2:
                continue

            _eval_checkpoint(
                checkpoint_path=current_ckpt,
                writer=writer,
                tok=tok,
                checkpoint_index=prev_ckpt_ind,
            )

    if writer is not None:
        try:
            writer.writer.close()
            del writer
        except Exception as e:
            logger.error(e)
    logger.info("END evaluate")


def _eval_checkpoint(
    checkpoint_path: str,
    writer,
    tok,
    checkpoint_index: int = 0, 
) -> None:
    logger.info(f"checkpoint_path: {checkpoint_path}")

    # initialize environment accrding to EVAL_DATASET
    if args.EVAL_DATASET == 'train':
        train_env = AirVLNENV(batch_size=args.batchSize, split='train', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_seen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_seen', tokenizer=tok)
    elif args.EVAL_DATASET == 'val_unseen':
        train_env = AirVLNENV(batch_size=args.batchSize, split='val_unseen', tokenizer=tok)
    elif args.EVAL_DATASET == 'test':
        train_env = AirVLNENV(batch_size=args.batchSize, split='test', tokenizer=tok)
    else:
        raise KeyError


    # make dir to save results
    EVAL_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/results/{}'.format(args.name, args.make_dir_time)
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if os.path.exists(fname):
        print("skipping -- evaluation exists.")
        return


    # make policy and load checkpoint
    trainer = VLNCETrainer(
        load_from_ckpt=True,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        ckpt_path=checkpoint_path,
    )
    trainer.policy.eval()

    if torch.cuda.is_available():
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()
    gc.collect()


    #
    stats_episodes = {}
    episodes_to_eval = len(train_env.data) # the number of eval data
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)

    with torch.no_grad():
        start_iter = 0
        end_iter = len(train_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, train_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * train_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            train_env.next_minibatch() # sampel minibatch and pass to env port
            if train_env.batch is None:
                logger.warning('train_env.batch is None, going to break and stop collect')
                break

            if args.policy_type in ['seq2seq', 'cma']:
                rnn_states = torch.zeros(
                    train_env.batch_size,
                    trainer.policy.net.num_recurrent_layers,
                    trainer.policy.net.state_encoder.hidden_size,
                    device=trainer.device,
                )
                prev_actions = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.long,
                    device=trainer.device,
                )
                not_done_masks = torch.zeros(
                    train_env.batch_size,
                    1,
                    dtype=torch.uint8,
                    device=trainer.device,
                )
            else:
                raise NotImplementedError

            rgb_frames = [[] for _ in range(train_env.batch_size)] # batch list of list

            episodes = [[] for _ in range(train_env.batch_size)]
            skips = [False for _ in range(train_env.batch_size)]
            dones = [False for _ in range(train_env.batch_size)]
            envs_to_pause = []

            outputs = train_env.reset() # return self.get_obs ( observations, reward, done, info)
            # preprocess observations
            observations, _, dones, _ = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, trainer.device) # "instruction"/"progress"/"teacher_action"/"pose"

            ended = False

            # rollout for maxAction steps and evaluation
            for t in range(int(args.maxAction)):
                logger.info('checkpoint_index:{} \t {} - {} / {} \t {}'.format(checkpoint_index, idx, t, end_iter, not_done_masks.cpu().numpy().reshape((-1,)).tolist()))

                actions, rnn_states = trainer.policy.act(
                    batch,
                    rnn_states,
                    prev_actions,
                    not_done_masks, # if done, mask is 0 episode is done
                    deterministic=True,
                    step=t,
                ) # the max prob action B  1 , length = 1 ,such as time rollout
                prev_actions.copy_(actions)

                # Make action and get the new state B action shape? B 1? list of batch
                actions = [temp[0] for temp in actions.cpu().numpy()]
                train_env.makeActions(actions) # step in envs and update_measurements env.sim_states[i].NDTW['_metric']

                outputs = train_env.get_obs() # update the env and return ( observations, reward, done, info)
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                batch = batch_obs(observations, trainer.device)

                logger.info('action: {}'.format(actions))

                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8,
                    device=trainer.device,
                )

                # for tttt in range(len(train_env.batch)):
                #     train_env.threading_lock_lmdb_features_txn.acquire()
                #     train_env.lmdb_features_txn.put(
                #         str('{}_{}_{}'.format(infos[tttt]['episode_id'], t, 'exp_1')).encode(),
                #         msgpack_numpy.packb(
                #             observations[tttt], use_bin_type=True
                #         ),
                #     )
                #     train_env.lmdb_features_txn.commit()
                #     train_env.lmdb_features_start_id = train_env.lmdb_features_env.stat()["entries"]
                #     train_env.lmdb_features_txn = train_env.lmdb_features_env.begin(write=True)
                #     train_env.threading_lock_lmdb_features_txn.release()
                #     logger.info('lmdb of {}, lmdb_start_id: {}'.format(train_env.split, train_env.lmdb_features_start_id))

                # reset envs and observations if necessary save video
                for i in range(train_env.batch_size):
                    if args.EVAL_GENERATE_VIDEO:
                        frame = observations_to_image(observations[i], infos[i]) # H W*2 3 RGB + DEPTH
                        frame = append_text_to_image(
                            frame, train_env.batch[i]['instruction']['instruction_text']
                        ) # raw image + instruction text (behind)
                        rgb_frames[i].append(frame) # add frame to list

                    if not dones[i] or skips[i]:
                        continue

                    skips[i] = True
                    pbar.update()

                if np.array(dones).all():
                    ended = True
                    break

            for t in range(int(train_env.batch_size)): # save info of each episode with json
                stats_episodes[str(train_env.batch[t]['episode_id'])] = infos[t]

                EVAL_SAVE_EVERY_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results_every/{}'.format(args.name, args.make_dir_time)
                if not os.path.exists(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index))):
                    os.makedirs(str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)), exist_ok=True)

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR / str(checkpoint_index)),
                    f"{train_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f)

                if args.EVAL_GENERATE_VIDEO:
                    EVAL_GENERATE_VIDEO_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/videos/{}'.format(args.name, args.make_dir_time)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=train_env.batch[t]['episode_id'],
                        checkpoint_idx=checkpoint_index,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]['ndtw'],
                        },
                        tb_writer=writer,
                    )

                logger.info((
                    'result-{} \t' +
                    'distance_to_goal: {} \t' +
                    'success: {} \t' +
                    'ndtw: {} \t' +
                    'sdtw: {} \t' +
                    'path_length: {} \t' +
                    'oracle_success: {} \t' +
                    'steps_taken: {}'
                ).format(
                    t,
                    infos[t]['distance_to_goal'],
                    infos[t]['success'],
                    infos[t]['ndtw'],
                    infos[t]['sdtw'],
                    infos[t]['path_length'],
                    infos[t]['oracle_success'],
                    infos[t]['steps_taken']
                )) # log the results of each episode

    # end
    pbar.close()


    # save and compute results good
    
    EVAL_INTERMEDIATE_RESULTS_DIR = Path(args.project_prefix) / 'DATA/output/{}/eval/intermediate_results/{}'.format(args.name, args.make_dir_time)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_INTERMEDIATE_RESULTS_DIR):
        os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f) # info of all episodes

    #
    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items(): # only save metric float
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = (
            sum(v[stat_key] for v in stats_episodes.values())
            / num_episodes
        )

    #
    fname = os.path.join(
        EVAL_RESULTS_DIR,
        f"stats_ckpt_{checkpoint_index}_{train_env.split}.json",
    )
    if not os.path.exists(EVAL_RESULTS_DIR):
        os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    with open(fname, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    checkpoint_num = checkpoint_index + 1
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")
        writer.add_scalar(f"eval_{train_env.split}_{k}", v, checkpoint_num)

    try:
        train_env.simulator_tool.closeScenes()
    except:
        pass


if __name__ == "__main__":
    setup()

    if args.run_type == 'collect':
        collect_data()
    elif args.run_type == 'train':
        train_vlnce()
    elif args.run_type == 'eval':
        eval_vlnce()
    else:
        raise NotImplementedError


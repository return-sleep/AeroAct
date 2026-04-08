import copy
import random
import time

import msgpack_numpy
import numpy as np
import math
from gym import spaces
import lmdb
import os
import json
from pathlib import Path
import airsim
import threading
from fastdtw import fastdtw
import tqdm
import sys
sys.path.append('/attached/remote-home2/xhl/9_UAV_VLN/NaVILA')
from typing import Dict, List, Optional

from AirVLN_src.common.param import args
from AirVLN_utils.logger import logger
from airsim_plugin.AirVLNSimulatorClientTool import AirVLNSimulatorClientTool
from airsim_plugin.airsim_settings import AirsimActions, AirsimActionSettings
from AirVLN_utils.env_utils import SimState, getPoseAfterMakeAction
from AirVLN_utils.env_vector import VectorEnvUtil
from AirVLN_utils.shorest_path_sensor import EuclideanDistance3


def load_my_datasets(splits):
    import random
    data = []
    vocab = {}
    old_state = random.getstate()
    for split in splits:
        components = split.split("@") # train@300 -> ['train', '300']
        number = -1
        if len(components) > 1:
            split, number = components[0], int(components[1])

        # Load Json   TODO add hyper to control the dataset aerialvln-s or aerialvln
        with open(f'AirVLN-Dataset/data/aerialvln-s/{split}.json', 'r', encoding='utf-8') as f:
            print(f'AirVLN-Dataset/data/aerialvln-s/{split}.json')
            new_data = json.load(f)
            new_data = new_data['episodes'] # 0,1,2 dict
        # Partition
        if number > 0:
            random.seed(1)              # Make the data deterministic, additive
            random.shuffle(new_data)
            new_data = new_data[:number] # fixed number of data

        # Join
        data += new_data
    random.setstate(old_state)      # Recover the state of the random generator
    return data, vocab # data is a list of dict, vocab is empty dict for now


class AirVLNENV: # only for evaluation
    def __init__(self, batch_size=8, split='train',
                 seed=1, tokenizer=None,
                 dataset_group_by_scene=True,
                 ):
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        if tokenizer:
            self.tok = tokenizer
        self.dataset_group_by_scene = dataset_group_by_scene

        load_data, vocab = load_my_datasets([split]) # load data from json file, dict 
        self.ori_raw_data = load_data.copy()
        logger.info('Loaded with {} instructions, using split: {}'.format(len(load_data), split))

        self.index_data = 0
        self.data = self.ori_raw_data
        if split =='train':
            random.shuffle(self.data)
        if args.EVAL_NUM != -1 and int(args.EVAL_NUM) > 0:
            [random.shuffle(self.data) for i in range(10)]
            self.data = self.data[:int(args.EVAL_NUM)].copy()  # only eval fixed num data
        if dataset_group_by_scene:
            self.data = self._group_scenes()
            logger.warning('dataset grouped by scene')

        scenes = [item['scene_id'] for item in self.data]
        self.scenes = set(scenes) # unique scenes id list

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(args.Image_Height_RGB, args.Image_Width_RGB, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=1, shape=(args.Image_Height_DEPTH, args.Image_Width_DEPTH, 1), dtype=np.float32),
            "instruction": spaces.Discrete(1), # why discrete?
            "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
        })
        self.action_space = spaces.Discrete(int(len(AirsimActions))) # ?
        # record current state and metric
        self.sim_states: Optional[List[SimState], List[None]] = [None for _ in range(batch_size)]
        self.last_scene_id_list = []
        self.one_scene_could_use_num = 5000
        self.this_scene_used_cnt = 0

        self.init_VectorEnvUtil()

    def _group_scenes(self):
        assert self.dataset_group_by_scene, 'error args param'

        scene_sort_keys: Dict[str, int] = {}
        for item in self.data:
            if str(item['scene_id']) not in scene_sort_keys:
                scene_sort_keys[str(item['scene_id'])] = len(scene_sort_keys)

        return sorted(self.data, key=lambda e: scene_sort_keys[str(e['scene_id'])])

    def init_VectorEnvUtil(self):
        self.delete_VectorEnvUtil()

        self.load_scenes = [int(_scene) for _scene in list(self.scenes)]
        self.VectorEnvUtil = VectorEnvUtil(self.load_scenes, self.batch_size)

    def delete_VectorEnvUtil(self):
        if hasattr(self, 'VectorEnvUtil'):
            del self.VectorEnvUtil

        import gc
        gc.collect()

    #
    def next_minibatch(self, skip_scenes=[], data_it=0):
        batch = []

        while True:
            if self.index_data >= len(self.data)-1: # > the number of data
                random.shuffle(self.data)
                logger.warning('random shuffle data')
                if self.dataset_group_by_scene:
                    self.data = self._group_scenes()
                    logger.warning('dataset grouped by scene')

                if len(batch) == 0:
                    self.index_data = 0
                    self.batch = None
                    return

                self.index_data = self.batch_size - len(batch) # shuffle the data again and get new batch
                batch += self.data[:self.index_data] # left sampled from new shuffled data
                break

            new_episode = self.data[self.index_data]

            # skip scenes by input skip_scenes
            if new_episode['scene_id'] in skip_scenes:
                self.index_data += 1
                continue

            batch.append(new_episode)
            self.index_data += 1

            if len(batch) == self.batch_size: # get a batch list 
                break

        self.batch = copy.deepcopy(batch)
        assert len(self.batch) == self.batch_size, 'next_minibatch error'

        self.VectorEnvUtil.set_batch(self.batch) # parallel set batch to VectorEnvUtil

    #
    def changeToNewEpisodes(self):
        self._changeEnv(need_change=False)

        self._setEpisodes()

        self.update_measurements()

    def _changeEnv(self, need_change: bool = True):
        try:
            scene_id_list = [item['scene_id'] for item in self.batch]
        except:
            print("AirVLN_src/vlnce_src/env.py", self.batch )
        assert len(scene_id_list) == self.batch_size, 'error'

        machines_info_template = copy.deepcopy(args.machines_info)
        total_max_scene_num = 0
        for item in machines_info_template:
            total_max_scene_num += item['MAX_SCENE_NUM']
        assert self.batch_size <= total_max_scene_num, 'error args param: batch_size'

        machines_info = []
        ix = 0
        for index, item in enumerate(machines_info_template):
            machines_info.append(item)
            delta = min(self.batch_size, item['MAX_SCENE_NUM'], len(scene_id_list)-ix)
            machines_info[index]['open_scenes'] = scene_id_list[ix : ix + delta]
            ix += delta

        #
        cnt = 0
        for item in machines_info:
            cnt += len(item['open_scenes'])
        assert self.batch_size == cnt, 'error create machines_info'

        #
        if self.this_scene_used_cnt < self.one_scene_could_use_num and \
                len(set(scene_id_list)) == 1 and len(set(self.last_scene_id_list)) == 1 and \
                scene_id_list[0] is not None and self.last_scene_id_list[0] is not None and scene_id_list[0] == self.last_scene_id_list[0] and \
                need_change == False:
            self.this_scene_used_cnt += 1
            logger.warning('no need to change env: {}'.format(scene_id_list))
            return
        else:
            logger.warning('to change env: {}'.format(scene_id_list))

        #
        while True:
            try:
                self.machines_info = copy.deepcopy(machines_info)
                # start render
                self.simulator_tool = AirVLNSimulatorClientTool(machines_info=self.machines_info)
                self.simulator_tool.run_call()
                break
            except Exception as e:
                logger.error("Failed to open scenes {}".format(e))
                time.sleep(3)
            except:
                logger.error('Failed to open scenes')
                time.sleep(3)

        self.last_scene_id_list = scene_id_list.copy()
        self.this_scene_used_cnt = 1

    def _setEpisodes(self):
        start_position_list = [item['start_position'] for item in self.batch]
        start_rotation_list = [item['start_rotation'] for item in self.batch]

        #
        poses = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                # fixed start position and rotation
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                poses[index_1].append(pose)
                cnt += 1

        result = self.simulator_tool.setPoses(poses=poses)
        if not result:
            logger.error('Failed to set poses')
            self.reset_to_this_pose(poses)

        #
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2, _ in enumerate(item['open_scenes']):
                pose = airsim.Pose(
                    position_val=airsim.Vector3r(
                        x_val=start_position_list[cnt][0],
                        y_val=start_position_list[cnt][1],
                        z_val=start_position_list[cnt][2],
                    ),
                    orientation_val=airsim.Quaternionr(
                        x_val=start_rotation_list[cnt][1],
                        y_val=start_rotation_list[cnt][2],
                        z_val=start_rotation_list[cnt][3],
                        w_val=start_rotation_list[cnt][0],
                    ),
                )
                self.sim_states[cnt] = SimState(index=cnt, step=0, episode_info=self.batch[cnt], pose=pose)
                self.sim_states[cnt].trajectory = [[
                    pose.position.x_val, pose.position.y_val, pose.position.z_val, # xyz
                    pose.orientation.x_val, pose.orientation.y_val, pose.orientation.z_val, pose.orientation.w_val, # xyzw
                ]]
                cnt += 1

    #
    def get_obs(self): # save rgb when  
        obs_states = (
            self._getStates()
        )  # list of SimState (_rgb_image, _depth_image, state) self.sim_states[cnt].is_end = True

        obs, states = self.VectorEnvUtil.get_obs(obs_states)
        self.sim_states = states

        return obs

    def _getStates(self):
        while True:
            responses = self.simulator_tool.getImageResponses(get_rgb=True, get_depth=True)
            if responses is None:
                poses = self._get_current_pose()
                self.reset_to_this_pose(poses)
                time.sleep(3)
            else:
                break

        #
        cnt = 0
        for item in responses:
            cnt += len(item)
        assert len(responses) == len(self.machines_info), 'error'
        assert cnt == self.batch_size, 'error'

        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                depth_image = responses[index_1][index_2][1]
                collision_sensor_result = (np.array(depth_image) < 0.004).sum() / np.array(depth_image).flatten().shape[0]
                if collision_sensor_result > 0.1:
                    self.sim_states[cnt].is_collisioned = True
                    self.sim_states[cnt].is_end = True
                    logger.warning('collisioned: {}'.format(cnt))

                cnt += 1

        #
        states = [None for _ in range(self.batch_size)]
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            for index_2 in range(len(item['open_scenes'])):
                rgb_image = responses[index_1][index_2][0]
                if rgb_image is not None:
                    _rgb_image = np.array(rgb_image)
                else:
                    _rgb_image = None

                depth_image = responses[index_1][index_2][1]
                if depth_image is not None:
                    _depth_image = np.array(depth_image)
                else:
                    _depth_image = None

                state = self.sim_states[cnt]

                states[cnt] = (_rgb_image, _depth_image, state)
                cnt += 1

        return states # list (_rgb_image, _depth_image, state)

    def _get_current_pose(self) -> list:
        poses = []

        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses[index_1].append(
                    self.sim_states[cnt].pose
                )
                cnt += 1

        return poses

    #
    def reset(self):
        self.changeToNewEpisodes()
        return self.get_obs()

    def reset_to_this_pose(self, poses):
        #
        self._changeEnv(need_change=True)

        result = self.simulator_tool.setPoses(poses=poses)
        if not result:
            logger.error('Failed to reset to this pose')
            self.reset_to_this_pose(poses)

    def makeActions(self, action_list): # len(action_list) == batch_size
        #
        poses = []
        for index, action in enumerate(action_list):
            if self.sim_states[index].is_end == True:
                action = AirsimActions.STOP
                # continue
            # maxAction = 500
            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            # predict stop action or max action
            state = self.sim_states[index]

            pose = copy.deepcopy(state.pose)
            new_pose = getPoseAfterMakeAction(pose, action) # update pose after action
            poses.append(new_pose) # list of pose? multi-env?

        poses_formatted = []
        cnt = 0
        for index_1, item in enumerate(self.machines_info):
            poses_formatted.append([])
            for index_2, _ in enumerate(item['open_scenes']):
                poses_formatted[index_1].append(poses[cnt])
                cnt += 1

        #
        result = self.simulator_tool.setPoses(poses=poses_formatted) # update poses in simulator
        if not result:
            logger.error('Failed to set poses')
            self.reset_to_this_pose(poses_formatted)

        #
        for index, action in enumerate(action_list): # action for multi-env index of env ids
            if self.sim_states[index].is_end == True:
                continue

            if action == AirsimActions.STOP or self.sim_states[index].step >= int(args.maxAction):
                self.sim_states[index].is_end = True

            self.sim_states[index].step += 1
            self.sim_states[index].pose = poses[index] # state  t+1 
            self.sim_states[index].trajectory.append([
                poses[index].position.x_val, poses[index].position.y_val, poses[index].position.z_val, # xyz
                poses[index].orientation.x_val, poses[index].orientation.y_val, poses[index].orientation.z_val, poses[index].orientation.w_val, # 
            ]) # trajectory t+1  7dim   different from pose ?
            self.sim_states[index].pre_action = action # action t

        # update measurement
        self.update_measurements() # update metrics

    # function to update metrics
    def update_measurements(self):
        self._update_DistanceToGoal()
        self._updata_Success()
        self._updata_NDTW()
        self._updata_SDTW()
        self._update_PathLength()
        self._update_OracleSuccess()
        self._update_StepsTaken()

    def _update_DistanceToGoal(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if self.sim_states[i].DistanceToGoal['_previous_position'] is None or \
                not np.allclose(self.sim_states[i].DistanceToGoal['_previous_position'], current_position, atol=1):
                distance_to_target = EuclideanDistance3(
                    np.array(current_position)[0:2],
                    np.array(state.episode_info['goals'][0]['position'])[0:2]
                )
                self.sim_states[i].DistanceToGoal['_previous_position'] = current_position
                self.sim_states[i].DistanceToGoal['_metric'] = distance_to_target

    def _updata_Success(self):
        for i, state in enumerate(self.sim_states):
            distance_to_target = self.sim_states[i].DistanceToGoal['_metric']
            if (
                self.sim_states[i].is_end # is key
                and distance_to_target <= self.sim_states[i].SUCCESS_DISTANCE
            ):
                self.sim_states[i].Success['_metric'] = 1.0
            else:
                self.sim_states[i].Success['_metric'] = 0.0

    def _updata_NDTW(self):
        def euclidean_distance(
                position_a,
                position_b,
        ) -> float:
            return np.linalg.norm(
                np.array(position_b) - np.array(position_a), ord=2
            )

        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if len(state.NDTW['locations']) == 0:
                self.sim_states[i].NDTW['locations'].append(current_position)
            else:
                if current_position.tolist() == state.NDTW['locations'][-1].tolist():
                    continue
                self.sim_states[i].NDTW['locations'].append(current_position)

            dtw_distance = fastdtw(
                self.sim_states[i].NDTW['locations'], self.sim_states[i].NDTW['gt_locations'], dist=euclidean_distance
            )[0]

            nDTW = np.exp(
                -dtw_distance / (len(self.sim_states[i].NDTW['gt_locations']) * self.sim_states[i].SUCCESS_DISTANCE)
            )
            self.sim_states[i].NDTW['_metric'] = nDTW

    def _updata_SDTW(self):
        for i, state in enumerate(self.sim_states):
            ep_success = self.sim_states[i].Success['_metric']
            nDTW = self.sim_states[i].NDTW['_metric']
            self.sim_states[i].SDTW['_metric'] = ep_success * nDTW

    def _update_PathLength(self):
        for i, state in enumerate(self.sim_states):

            current_position = np.array([
                state.pose.position.x_val,
                state.pose.position.y_val,
                state.pose.position.z_val
            ])

            if state.PathLength['_previous_position'] is None:
                self.sim_states[i].PathLength['_previous_position'] = current_position

            self.sim_states[i].PathLength['_metric'] += EuclideanDistance3(
                current_position, self.sim_states[i].PathLength['_previous_position']
            )
            self.sim_states[i].PathLength['_previous_position'] = current_position

    def _update_OracleSuccess(self):
        for i, state in enumerate(self.sim_states):
            d = self.sim_states[i].DistanceToGoal['_metric']
            self.sim_states[i].OracleSuccess['_metric'] = float(
                self.sim_states[i].OracleSuccess['_metric'] or d <= self.sim_states[i].SUCCESS_DISTANCE
            )

    def _update_StepsTaken(self):
        for i, state in enumerate(self.sim_states):
            self.sim_states[i].StepsTaken['_metric'] = self.sim_states[i].step

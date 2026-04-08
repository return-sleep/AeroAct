import os
import re
import math
import json
import torch
import copy
import tqdm
import time
import numpy as np
from PIL import Image
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import KeywordsStoppingCriteria, process_images, tokenizer_image_token
from llava.conversation import SeparatorStyle, conv_templates
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def sample_and_pad_images(images, num_frames=8, width=448, height=448, data_args = None):
    """Sample and pad images to a fixed number of frames.

    Args:
        images (list of PIL.Image): List of input images.
        num_frames (int): Desired number of frames.
        width (int): Width of the padded images.
        height (int): Height of the padded images.
    Returns:
        list of PIL.Image: List of sampled and padded images.
    """
    frames = copy.deepcopy(images)
    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.insert(0, Image.new("RGB", (width, height), color=(0, 0, 0)))
        return frames
    latest_frame = frames[-1]
    sampled_indices = np.linspace(0, len(frames) - 1, num=num_frames - 1, endpoint=False, dtype=int)
    sampled_frames = [frames[i] for i in sampled_indices] + [latest_frame]
    return sampled_frames



def convert_text_to_action(outputs):
    """Convert text-form action to action list.
    Args:
        outputs (str): Text-form action output from the model. 
    Returns:
        list of int: List of action integers.
    Example:
        outputs = "The next action is moving forward 15 meters."
        actions = convert_text_to_action(outputs)
        print(actions)  # [1, 1, 1]
    """
    patterns = {
        0: re.compile(r"\bstop\b", re.IGNORECASE),
        1: re.compile(r"\bis move forward\b", re.IGNORECASE),
        2: re.compile(r"\bis turn left\b", re.IGNORECASE),
        3: re.compile(r"\bis turn right\b", re.IGNORECASE),
        4: re.compile(r"\bis ascend\b", re.IGNORECASE),
        5: re.compile(r"\bis descend\b", re.IGNORECASE),
        6: re.compile(r"\bis move left\b", re.IGNORECASE),
        7: re.compile(r"\bis move right\b", re.IGNORECASE),
    }
    # action unit
    primitive_map = {
        1: 5,  # move forward: 5 units
        2: 15,  # turn left:   15 degrees
        3: 15,  # turn right:  15 degrees
        4: 2,  # ascend:      2 units
        5: 2,  # descend:     2 units
        6: 5,  # move left:   5 units
        7: 5,  # move right:  5 units
    }

    # Function to map a string to an action integer, not split 75cm to 3*25 cm
    def map_string_to_action(s):
        for action, pattern in patterns.items():
            if pattern.search(s):
                return action  # patterns id
        return None  # Return None if no match is found

    try:
        actions_rough = map_string_to_action(outputs)
        if actions_rough == 0:
            actions = [0]
        else:
            num_part_match = re.search(r"\d+", outputs)
            if num_part_match:
                num_part = int(num_part_match.group())
                base_unit = primitive_map[actions_rough]
                num_actions = max(1, int(math.ceil(num_part / base_unit)))
                actions = [actions_rough] * num_actions
            else:
                actions = [actions_rough]
        return actions
    except:
        actions = [3]  # default to turn right if parsing fails, to avoid risky forward 
        return actions


def preprocess_obs(outputs):
    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
    instruction = observations[0]["instruction"]  # string
    curr_rgb = observations[0]["rgb"]  # current rgb image
    curr_rgb = curr_rgb[..., ::-1]  # BGR -> RGB
    curr_rgb = Image.fromarray(curr_rgb, mode="RGB")
    curr_depth = curr_rgb
    return instruction, curr_rgb, curr_depth, observations, infos, dones


def get_text_action(
    past_and_current_rgbs,
    instruction,
    tokenizer,
    model,
    image_processor,
    conv_mode="llama_3", # "auto" for 2B model, "llama_3" for 8B model
    temperature=0.0, # 0.0 for deterministic, higher for more random
    timestep=0,
    history_queue_len=0,
    temporal_aggregation_frames = None, 
    data_args = None
):
    """
    Get text-form action from the model given past and current RGB images and instruction.
    Args:
        past_and_current_rgbs (list of PIL.Image): List of past and current RGB images
        instruction (str): Navigation instruction.
        tokenizer: Tokenizer for the model.
        model: The multimodal model.
        image_processor: Image processor for the model.
        conv_mode (str): Conversation mode, either "llama_3" or "auto".
        temperature (float): Sampling temperature for generation.
        timestep (int): Current timestep (not used in prompt).
        history_queue_len (int): Number of past frames to consider.
        temporal_aggregation_frames (int or None): Number of frames for temporal aggregation.
    Returns:
        str: Generated text-form action.
    """
    num_video_frames = model.config.num_video_frames
    model_dtype = getattr(model, "dtype", torch.float16)
    device = next(model.parameters()).device
    past_and_current_rgbs = past_and_current_rgbs[-history_queue_len:] # memory bank
    print("valid image num: (memory bank)", len(past_and_current_rgbs))
    past_and_current_rgbs = sample_and_pad_images(
        past_and_current_rgbs, num_frames=num_video_frames, data_args = data_args
    )  # return list, len = num_video_frames
    print("sampled and padded image num:", len(past_and_current_rgbs))
   
    # construct question prompt
    interleaved_images = "<image>\n" * (len(past_and_current_rgbs) - 1)  # history views
    if temporal_aggregation_frames: # temporal_pooling setting
        interleaved_images = "<image>\n" * (temporal_aggregation_frames -  1)
    
    question = (
            f"Imagine you are a robot programmed for navigation tasks. You have been given a video "
            f"of historical observations {interleaved_images}, and the current observation <image>.\n"
            f'Your assigned task is: "{instruction}".'
            f"Analyze this series of images to decide your next action, which could be turning left or right by a specific "
            f"degree, moving forward, right or left a certain distance, ascending or descending vertically, or stop if the task is completed."
        )
    # get prompt when version is llama_3
    if conv_mode == "llama_3": # for 8B model
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()  # error when version is auto
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]  # ['<|eot_id|>'] tok_len= 7 for 2B model,  sep: '<|eot_id|>' tok_len= 1 for 8B model
    elif conv_mode == "auto": # for 2B model refer to llava/utils/tokenizer.py Line 101
        conv = conv_templates["auto"].copy()
        messages = [
            {"from": "human", "value": question},
            # {"from": "gpt", "value": None}, # why not add get
        ]
        conversation = []
        for m in messages:
            message = {}
            if m["from"] == "human":
                message["role"] = "user"
            elif m["from"] == "gpt":
                message["role"] = "assistant"
            else:
                raise ValueError(f"Unexpected sender '{m['from']}' in conversation entry.")

            message["content"] = m["value"]
            conversation.append(message)  # {"role": "human", "content": question}

        prompt = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        stop_str = tokenizer.eos_token
        stop_str_sep = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str, stop_str_sep]  # ['<eot>'] sep: '<eot>' tok_len= 1 for 2B/8B model

    # tokenize prompt and images
    images_tensor = process_images(past_and_current_rgbs, image_processor, model.config).to(
        device=device, dtype=model_dtype
    )
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        # llava/model/llava_arch.py def generate()
        output_ids = model.generate(
            input_ids,
            images=images_tensor.half(),
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature,
            max_new_tokens=32,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs

if __name__ == "__main__":
    from AirVLN.src.common.param import args
    from AirVLN.src.vlnce_src.env import AirVLNENV
    from AirVLN.utils.logger import logger
    from AirVLN.utils.utils import append_text_to_image, observations_to_image, generate_video

    temperature = args.inference_sample_temperature
    # step 1: make model
    model_path = args.model_path
    model_name = args.model_path.split("/")[1]  # is lora in model_name? if full-tuning model_base = None
    model_base = "a8cheng/navila-siglip-llama3-8b-v1.5-pretrain" if "lora" in model_name.lower() else None  #
    logger.info("Load weight from {}".format(model_path))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_name, model_base, device="cuda", device_map=None
    )
    model.eval()
    model = model.to(device="cuda", dtype=torch.float16)

    if  "nvila" in model_name.lower():
        conv_mode = "llama_3"  # for NaVILA-8B model
    else:
        conv_mode = "auto"  # for NVILA model 
    print("model_path", model_path, "conv_mode", conv_mode)
    
    data_args_path = os.path.join(model_path, "data_args.json")
    data_args = json.load(open(data_args_path, "r"))
    idx_type = data_args.get("idx_type", "default")
    history_queue_len = data_args.get("hisotry_queue_len", 0) # fixed 
    temporal_aggregation_frames = data_args.get("temporal_aggregation_frames", None)
    sampling_type =  data_args.get("sampling_type", "default")
    K_recent =  data_args.get("K_recent", 1)
    print("idx_type:", idx_type, "history_queue_len:", history_queue_len, "temporal_aggregation_frames:", temporal_aggregation_frames)
    print('sampling_type', sampling_type, 'K_recent',K_recent)
    # step 2: make env
    dataset = args.dataset  # 'aerialvln-s' or 'aerialvln'
    eval_env = AirVLNENV(batch_size=1, split=args.split, dataset=dataset)
    # args.EVAL_GENERATE_VIDEO = True
    args.name = os.path.join(args.name, "_".join(args.model_path.split("/")[1:]), "temp{}".format(temperature))
    args.name = os.path.join(args.name,'debug') # args.name = debug for quick test, remove it for full evaluation 'default
    if dataset == 'aerialvln':
         EVAL_SAVE_EVERY_RESULTS_DIR = "AirVLN_eval_full/{}/{}/intermediate_results_every".format(args.split, args.name)
    else:
        EVAL_SAVE_EVERY_RESULTS_DIR = "AirVLN_eval/{}/{}/intermediate_results_every".format(args.split, args.name)
    os.makedirs(EVAL_SAVE_EVERY_RESULTS_DIR, exist_ok=True)
    print("make log file", EVAL_SAVE_EVERY_RESULTS_DIR)
    
    # step 3: eval
    start_time = time.time()
    # ============= eval ===================
    stats_episodes = {}
    episodes_to_eval = len(eval_env.data)
    pbar = tqdm.tqdm(total=episodes_to_eval, dynamic_ncols=True)
    with torch.no_grad():
        start_iter = 0
        end_iter = len(eval_env.data)
        cnt = 0
        for idx in range(start_iter, end_iter, eval_env.batch_size):
            if args.EVAL_NUM != -1 and cnt * eval_env.batch_size >= args.EVAL_NUM:
                break
            cnt += 1

            eval_env.next_minibatch()
            
            # import ipdb; ipdb.set_trace()
            rgb_frames = [[] for _ in range(eval_env.batch_size)]
            skips = [False for _ in range(eval_env.batch_size)]
            dones = [False for _ in range(eval_env.batch_size)]

            step = 0
            past_rgbs = [[]]  # len(past_rgbs)=bs=1
            # reset env and get first observation
            try:
                outputs = eval_env.reset()
                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR),
                    f"{eval_env.batch[0]['episode_id']}.json")
                if os.path.exists(f_intermediate_result_name):
                    print(eval_env.batch[0]['episode_id'], 'Alreadly Measure')
                    continue
            except:
                print(f"finish {idx} episode evaluation, but error in env.reset")
                break
            instruction, curr_rgb, curr_depth, observations, infos, dones = preprocess_obs(outputs)
            print(instruction)
            
            # interact with env util stop
            last_action = None
            consec = 0
            turn_consec = 0
            max_action_consec = 40 # for airvln-s & airvln
            max_turn_consec = 30
            action_record = []
            while step < args.maxAction + 1:
                # 1. make policy get (text-form) action
                past_and_current_rgbs = past_rgbs[0] + [curr_rgb]  # list of PIL.Image
                action_text = get_text_action(
                    past_and_current_rgbs,
                    instruction,
                    tokenizer,
                    model,
                    image_processor,
                    conv_mode,
                    temperature=temperature,
                    history_queue_len=history_queue_len,
                    temporal_aggregation_frames=temporal_aggregation_frames,
                    data_args=data_args
                )
                actions = convert_text_to_action(action_text)  # translate text-form action  to action list
                print("step", step, "action_text:", action_text, "actions:", actions)
                action_record.extend(actions)
                # 2. deploy action interact with env, e.g. actions =[1,1,1,1,2]
                action_num = len(actions)
                for i in range(action_num):
                    # update past_rgb queue
                    if idx_type == "default":
                        past_rgbs[0].append(curr_rgb)  # save step-wise rgb
                    elif idx_type == "merge" and i == 0:
                        past_rgbs[0].append(curr_rgb)  # save keyframe-wise rgb

                    # interact with env and get new observation
                    action = [actions[i]]  # [1]
                    eval_env.makeActions(action)
                    outputs = eval_env.get_obs()
                    instruction, curr_rgb, curr_depth, observations, infos, dones = preprocess_obs(outputs)

                    step += 1  # update action num
                    if actions[i] == last_action:
                        consec += 1
                    else:
                        last_action = actions[i]
                        consec = 1

                    if actions[i] in [2, 3]:
                        turn_consec += 1
                    else:
                        turn_consec = 0

                    if (
                        consec >= max_action_consec) or (turn_consec
                     >= max_turn_consec):  # avoid meaningless forward in the sky or sea apply stop
                        print(
                            f"step {step} force stop the episode due to too many ({consec}) consecutive same actions {actions[i]} or turns {turn_consec} timesteps"
                        )
                        action = [0]  # [1]
                        eval_env.makeActions(action)
                        outputs = eval_env.get_obs()
                        instruction, curr_rgb, curr_depth, observations, infos, dones = preprocess_obs(outputs)
                    
                    # print info metrics
                    print(
                        step,
                        infos[0]["done"],  # do you stop?
                        infos[0]["distance_to_goal"],
                        infos[0]["success"],
                        infos[0]["oracle_success"],
                        infos[0]["is_collisioned"], # rotation will cause collision?
                    )
                    # save video frames and update pbar
                    for i in range(eval_env.batch_size):
                        if args.EVAL_GENERATE_VIDEO:  # save video
                            frame = observations_to_image(observations[i], infos[i])  # rgb of i-sample
                            frame = append_text_to_image(frame, eval_env.batch[i]["instruction"]["instruction_text"])
                            rgb_frames[i].append(frame)

                        if not dones[i] or skips[i]:
                            continue
                        skips[i] = True
                        pbar.update()

                    if np.array(dones).all():  # episode finished
                        break

                if np.array(dones).all():  # batch episode finished
                    break

            # save metric at each episode
            for t in range(int(eval_env.batch_size)):
                stats_episodes[str(eval_env.batch[t]["episode_id"])] = infos[t]

                f_intermediate_result_name = os.path.join(
                    str(EVAL_SAVE_EVERY_RESULTS_DIR),
                    f"{eval_env.batch[t]['episode_id']}.json",
                )
                f_intermediate_trajectory = {**infos[t]}
                f_intermediate_trajectory["action_record"] = action_record
                with open(f_intermediate_result_name, "w") as f:
                    json.dump(f_intermediate_trajectory, f, indent=4)

                if args.EVAL_GENERATE_VIDEO:
                    if dataset == 'aerialvln':
                        EVAL_GENERATE_VIDEO_DIR = "AirVLN_eval_full/{}/{}/videos".format(args.split, args.name)
                    else:
                        EVAL_GENERATE_VIDEO_DIR = "AirVLN_eval/{}/{}/videos".format(args.split, args.name)
                    os.makedirs(EVAL_GENERATE_VIDEO_DIR, exist_ok=True)
                    generate_video(
                        video_option=["disk"],
                        video_dir=str(EVAL_GENERATE_VIDEO_DIR),
                        images=rgb_frames[t],
                        episode_id=eval_env.batch[t]["episode_id"],
                        checkpoint_idx=0,
                        metrics={
                            # "spl": infos[t]['spl'],
                            "ndtw": infos[t]["ndtw"],
                        },
                    )

                logger.info("{}-of-{}-episodes".format(i, len(eval_env.batch)))
                logger.info(
                    (
                        "result-{} \t"
                        + "distance_to_goal: {} \t"
                        + "success: {} \t"
                        + "ndtw: {} \t"
                        + "sdtw: {} \t"
                        + "path_length: {} \t"
                        + "oracle_success: {} \t"
                        + "steps_taken: {}"
                    ).format(
                        t,
                        infos[t]["distance_to_goal"],
                        infos[t]["success"],
                        infos[t]["ndtw"],
                        infos[t]["sdtw"],
                        infos[t]["path_length"],
                        infos[t]["oracle_success"],
                        infos[t]["steps_taken"],
                    )
                )
        # end
    pbar.close()

    # ================ average stats over episodes ================
    if dataset == 'aerialvln':
        EVAL_INTERMEDIATE_RESULTS_DIR = "AirVLN_eval_full/{}/{}/intermediate_results".format(args.split, args.name)
    else:
        EVAL_INTERMEDIATE_RESULTS_DIR = "AirVLN_eval/{}/{}/intermediate_results".format(args.split, args.name)
    f_intermediate_name = os.path.join(
        EVAL_INTERMEDIATE_RESULTS_DIR,
        f"stats_{eval_env.split}.json",
    )
    os.makedirs(EVAL_INTERMEDIATE_RESULTS_DIR, exist_ok=True)
    with open(f_intermediate_name, "w") as f:
        json.dump(stats_episodes, f, indent=4)

    new_stats_episodes = {}
    for i, j in stats_episodes.items():
        temp_1 = {}
        temp_1 = j.copy()

        temp_2 = temp_1.copy()
        for _i, _j in temp_2.items():
            if type(_j) == str or type(_j) == list or type(_j) == dict:
                del temp_1[_i]

        new_stats_episodes[i] = temp_1.copy()
    stats_episodes = new_stats_episodes.copy()

    aggregated_stats = {}
    num_episodes = len(stats_episodes)
    for stat_key in next(iter(stats_episodes.values())).keys():
        aggregated_stats[stat_key] = sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
    
    if dataset == 'aerialvln':
        EVAL_RESULTS_DIR = "AirVLN_eval_full/{}/{}/results".format(args.split, args.name)
    else:
        EVAL_RESULTS_DIR = "AirVLN_eval/{}/{}/results".format(args.split, args.name)
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    metric_name = os.path.join(
        EVAL_RESULTS_DIR,
        f"metric_{eval_env.split}.json",
    )
    with open(metric_name, "w") as f:
        json.dump(aggregated_stats, f, indent=4)

    logger.info(f"Episodes evaluated: {num_episodes}")
    for k, v in aggregated_stats.items():
        logger.info(f"Average episode {k}: {v:.6f}")

    try:
        eval_env.simulator_tool.closeScenes()
    except:
        pass

    end_time = time.time()
    print("eval total time:", end_time - start_time)

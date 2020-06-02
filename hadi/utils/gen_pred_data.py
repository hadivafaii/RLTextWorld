import os
from os.path import join as pjoin
import re
import spacy
import numpy as np
from tqdm import tqdm
import gym
import textworld.gym
from textworld import EnvInfos


def _get_traj_so_far(game_path, num_steps, tokenizer):
    requested_infos = EnvInfos(admissible_commands=True, game=True)
    env_id = textworld.gym.register_games(
        game_path,
        batch_size=None,
        request_infos=requested_infos,
        asynchronous=False,
        max_episode_steps=100)
    env = gym.make(env_id)
    obs, infos = env.reset()

    traj = ['[OBS]'] + preproc(obs, tokenizer)
    walkthrough = infos['game'].walkthrough

    for i in range(num_steps):
        obs, _, _, infos = env.step(walkthrough[i])
        traj.extend(['[ACT]'] + preproc(walkthrough[i], tokenizer) +
                    ['[OBS]'] + preproc(obs, tokenizer))

    current_adm_cmds = infos['admissible_commands']
    current_adm_cmds.remove("look")
    branch_size = len(current_adm_cmds)

    env_id = textworld.gym.register_games(
        game_path * branch_size,
        batch_size=branch_size,
        request_infos=requested_infos,
        asynchronous=False,
        max_episode_steps=100)

    envs = gym.make(env_id)
    envs.reset()

    for i in range(num_steps):
        cmds = [walkthrough[i]] * branch_size
        _ = envs.step(cmds)

    return envs, traj, current_adm_cmds


def _create_pred_data(env, traj_so_far, current_adm_cmds, walkthrough_cmd, tokenizer):
    if walkthrough_cmd not in current_adm_cmds:
        return None, False

    it_is_done = False
    outputs = ()

    next_states = []
    walkthrough_obs = None

    obs, _, done, infos = env.step(current_adm_cmds)

    for cc, dd, oo in zip(current_adm_cmds, done, obs):
        if cc == walkthrough_cmd:
            walkthrough_obs = oo
            if dd:
                it_is_done = True
        else:
            if "go" in walkthrough_cmd and "go" in cc:
                continue
            else:
                next_states.append(oo)

    obs_pred_trajs = []
    obs_pred_labels = []
    for fake_obs in next_states:
        obs_pred_trajs.append(
            traj_so_far
            + ['[ACT]'] + preproc(walkthrough_cmd, tokenizer)
            + ['[OBS]'] + preproc(fake_obs, tokenizer)
        )
        obs_pred_labels.append(0)

    obs_pred_trajs.append(
        traj_so_far
        + ['[ACT]'] + preproc(walkthrough_cmd, tokenizer)
        + ['[OBS]'] + preproc(walkthrough_obs, tokenizer)
    )
    obs_pred_labels.append(1)

    assert len(obs_pred_trajs) == len(obs_pred_labels), "must have same lengths"
    outputs += ([obs_pred_trajs, obs_pred_labels],)

    act_pred_trajs = []
    act_pred_labels = []
    for cmd in current_adm_cmds:
        if "go" in walkthrough_cmd and "go" in cmd and cmd != walkthrough_cmd:
            continue
        act_pred_trajs.append(
            traj_so_far
            + ['[ACT]'] + preproc(cmd, tokenizer)
            + ['[OBS]'] + preproc(walkthrough_obs, tokenizer)
        )
        if cmd == walkthrough_cmd:
            act_pred_labels.append(1)
        else:
            act_pred_labels.append(0)

    assert len(act_pred_trajs) == len(act_pred_labels), "must have same lengths"
    outputs += ([act_pred_trajs, act_pred_labels],)

    return outputs, it_is_done


def extract_pred_data(game_path, tokenizer):
    assert isinstance(game_path, str), "Must be path to an individual game"
    game_path = [game_path]

    requested_infos = EnvInfos(admissible_commands=True, game=True)
    env_id = textworld.gym.register_games(game_path, request_infos=requested_infos, max_episode_steps=100)
    env = gym.make(env_id)

    _, infos = env.reset()
    walkthrough = infos['game'].walkthrough

    obs_pred_data = []
    act_pred_data = []

    done = False
    steps_so_far = 0

    while not done:
        envs, traj_so_far, current_adm_cmds = _get_traj_so_far(game_path, steps_so_far, tokenizer)
        assert envs.batch_size == len(current_adm_cmds), "otherwise sth wrong"

        if steps_so_far >= len(walkthrough):
            # print("True/False for done kind of weridness encountered, moving on")
            break

        data, is_it_done = _create_pred_data(
            env=envs,
            traj_so_far=traj_so_far,
            current_adm_cmds=current_adm_cmds,
            walkthrough_cmd=walkthrough[steps_so_far],
            tokenizer=tokenizer)

        if data is not None:
            obs_pred_data.append(list(zip(*data[0])))
            act_pred_data.append(list(zip(*data[1])))
        # else:
            # print("Step. {}, weirdness encountered, moving on".format(steps_so_far))

        done = is_it_done
        steps_so_far += 1

    return obs_pred_data, act_pred_data


def process_pred_data(obs_pred_data, act_pred_data, w2i, data_config, config):
    assert len(obs_pred_data) == len(act_pred_data), "otherwise sth wrong"

    num_steps = len(obs_pred_data)
    max_len = data_config.max_len - 1

    act_pred = np.empty((0, max_len), dtype=int)
    obs_pred = np.empty((0, max_len), dtype=int)

    act_lbls = []
    obs_lbls = []

    for s in range(num_steps):
        current_data_obs = obs_pred_data[s]
        current_data_act = act_pred_data[s]

        for i in range(len(current_data_act)):
            act_lbls.append(current_data_act[i][1])
            data_indxs = [w2i[x] for x in current_data_act[i][0]]

            if len(data_indxs) <= max_len:
                padded_data_indxs = np.pad(data_indxs, (0, max_len - len(data_indxs)), constant_values=0)
            else:
                start_indices = np.where(np.array(data_indxs) == config.obs_id)[0]
                start_index = start_indices[np.where((len(data_indxs) - start_indices) < max_len)[0][0]]
                data_indxs = data_indxs[start_index:]
                padded_data_indxs = np.pad(data_indxs, (0, max_len - len(data_indxs)), constant_values=0)

            act_pred = np.concatenate([act_pred, np.expand_dims(padded_data_indxs, 0)])

        for i in range(len(current_data_obs)):
            obs_lbls.append(current_data_obs[i][1])
            data_indxs = [w2i[x] for x in current_data_obs[i][0]]

            if len(data_indxs) <= max_len:
                padded_data_indxs = np.pad(data_indxs, (0, max_len - len(data_indxs)), constant_values=0)
            else:
                start_indices = np.where(np.array(data_indxs) == config.obs_id)[0]
                start_index = start_indices[np.where((len(data_indxs) - start_indices) < max_len)[0][0]]
                data_indxs = data_indxs[start_index:]
                padded_data_indxs = np.pad(data_indxs, (0, max_len - len(data_indxs)), constant_values=0)

            obs_pred = np.concatenate([obs_pred, np.expand_dims(padded_data_indxs, 0)])

    column_mat = np.ones((len(act_pred), 1), dtype=int) * config.traj_id
    act_pred = np.concatenate([column_mat, act_pred], axis=1)

    column_mat = np.ones((len(obs_pred), 1), dtype=int) * config.traj_id
    obs_pred = np.concatenate([column_mat, obs_pred], axis=1)

    assert act_pred.shape[-1] == obs_pred.shape[-1] == data_config.max_len, "otherwise sth wrong"

    return (obs_pred, obs_lbls), (act_pred, act_lbls)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("game_type", help="(str) game type. (e.g. tw_cooking/train)", type=str)
    parser.add_argument("iter", help="iteration step", type=int)
    parser.add_argument("num_groups", help="how many subgroups to divide to", type=int)
    parser.add_argument("--game_spec", help="game specifics. default: None", type=str, default="",)

    args = parser.parse_args()

    import sys
    sys.path.append("..")
    from model.preprocessing import get_tokenizer, preproc
    from model.configuration import TransformerConfig, DataConfig
    from .gen_pretrain_data import save_data, compute_type_position_ids

    data_config = DataConfig(
        pretrain_modes=['ACT_PRED', 'OBS_PRED'],
        game_type=args.game_type,
        game_spec=args.game_spec,
        max_len=512,
        eps=[1.00],
        train_valid_test=True)
    config = TransformerConfig()

    if args.game_type.split('/')[1] == 'train':
        load_dir = data_config.games_dirs[0]
        save_dir = data_config.base_dirs[0]
        assert 'train' in load_dir and 'train' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'valid':
        load_dir = data_config.games_dirs[1]
        save_dir = data_config.base_dirs[1]
        assert 'valid' in load_dir and 'valid' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'test':
        load_dir = data_config.games_dirs[2]
        save_dir = data_config.base_dirs[2]
        assert 'test' in load_dir and 'test' in save_dir, "..."
    else:
        raise ValueError("Invalid game type entered (e.g. tw_cooking/train is correct)")
    os.makedirs(save_dir, exist_ok=True)

    lang_dir_file = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(data_config.max_len))
    lang_data_all = np.load(lang_dir_file, allow_pickle=True).item()
    current_lang_data = lang_data_all['eps={:.2f}'.format(data_config.epsilons[0])]

    game_files = os.listdir(load_dir)
    game_files = [pjoin(load_dir, g) for g in game_files if '.ulx' in g]

    group_size = int(np.ceil(len(game_files) / args.num_groups))

    a = args.iter * group_size
    b = (args.iter + 1) * group_size

    game_files = game_files[a:b]
    num_games = len(game_files)

    tokenizer = get_tokenizer()

    data_all = []
    for game in tqdm(game_files):
        # extract data
        obs_pred_data, act_pred_data = extract_pred_data(game, tokenizer)
        data_all.append([obs_pred_data, act_pred_data])

    save_ = pjoin(save_dir, 'raw_pred_data')
    os.makedirs(save_, exist_ok=True)
    file_name = 'iter={:d}.npy'.format(args.iter)
    np.save(pjoin(save_, file_name), data_all)

    print('Done!')

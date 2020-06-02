import os
from os.path import join as pjoin
import re
import spacy
import numpy as np
from tqdm import tqdm
import gym
import textworld.gym
from textworld import EnvInfos


def update_language(data_list, w2i):
    for x in data_list:
        for current_data in x:
            for i in range(len(current_data)):
                for word in current_data[i][0]:
                    if word not in w2i:
                        w2i.update({word: len(w2i)})
    i2w = {w2i[tok]: tok for tok in w2i}
    return w2i, i2w


def _get_traj_so_far3(env_id, num_steps, tokenizer):
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
    envs = [gym.make(env_id) for _ in range(len(current_adm_cmds))]
    _ = [e.reset() for e in envs]

    for i in range(num_steps):
        _ = [e.step(walkthrough[i]) for e in envs]

    return envs, traj, current_adm_cmds


def _get_traj_so_far(game_path, num_steps, tokenizer):
    requested_infos = EnvInfos(admissible_commands=True, game=True)
    env_id = textworld.gym.register_games(
        [game_path],
        batch_size=None,
        request_infos=requested_infos,
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
        [game_path] * branch_size,
        batch_size=branch_size,
        request_infos=requested_infos,
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


def _create_pred_data3(envs, traj_so_far, current_adm_cmds, walkthrough_cmd, tokenizer):
    if walkthrough_cmd not in current_adm_cmds:
        return None, False

    it_is_done = False
    outputs = ()

    next_states = []
    walkthrough_obs = None
    for e, cmd in zip(envs, current_adm_cmds):
        obs, _, done, infos = e.step(cmd)
        if cmd == walkthrough_cmd:
            walkthrough_obs = obs
            if done:
                it_is_done = True
        else:
            if "go" in walkthrough_cmd and "go" in cmd:
                continue
            else:
                next_states.append(obs)

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

    requested_infos = EnvInfos(admissible_commands=True, game=True)
    env_id = textworld.gym.register_games([game_path], request_infos=requested_infos, max_episode_steps=100)
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

    parser.add_argument(
        "game_type", help="(str) game type. (e.g. tw_cooking/train)",
        type=str
    )
    parser.add_argument(
        "--game_spec", help="game specifics such as brief or detailed goal, quest length and so on. default is None",
        type=str, default="",
    )

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
        save_dir = data_config.pretrain_dirs[0]
        assert 'train' in load_dir and 'train' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'valid':
        load_dir = data_config.games_dirs[1]
        save_dir = data_config.pretrain_dirs[1]
        assert 'valid' in load_dir and 'valid' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'test':
        load_dir = data_config.games_dirs[2]
        save_dir = data_config.pretrain_dirs[2]
        assert 'test' in load_dir and 'test' in save_dir, "..."
    else:
        raise ValueError("Invalid game type entered (e.g. tw_cooking/train is correct)")
    os.makedirs(save_dir, exist_ok=True)

    lang_dir_file = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(data_config.max_len))
    lang_data_all = np.load(lang_dir_file, allow_pickle=True).item()
    current_lang_data = lang_data_all['eps={:.2f}'.format(data_config.epsilons[0])]

    game_files = os.listdir(load_dir)
    game_files = [pjoin(load_dir, g) for g in game_files if '.ulx' in g]
    # TODO: temporary
    game_files = game_files[:5]

    tokenizer = get_tokenizer()

    obs_pred_all = []
    obs_lbls_all = []
    act_pred_all = []
    act_lbls_all = []
    for game in tqdm(game_files):
        # extract data
        obs_pred_data, act_pred_data = extract_pred_data(game, tokenizer)

        # load, update, and save language data
        lang_data_all = np.load(lang_dir_file, allow_pickle=True).item()
        current_lang_data = lang_data_all['eps={:.2f}'.format(data_config.epsilons[0])]
        w2i, i2w = update_language(data_list=[obs_pred_data, act_pred_data], w2i=current_lang_data['w2i'])
        current_lang_data['w2i'] = w2i
        current_lang_data['i2w'] = i2w
        np.save(lang_dir_file, {'eps={:.2f}'.format(data_config.epsilons[0]): current_lang_data})

        # process the extracted data using updated language
        (obs_pred, obs_lbls), (act_pred, act_lbls) = process_pred_data(obs_pred_data, act_pred_data,
                                                                       w2i, data_config, config)

        obs_pred_all.append(obs_pred)
        obs_lbls_all.extend(obs_lbls)
        act_pred_all.append(act_pred)
        act_lbls_all.extend(act_lbls)

    obs_token_ids = np.concatenate(obs_pred_all)
    obs_type_ids, obs_position_ids = compute_type_position_ids(obs_token_ids, config)
    obs_labels = np.array(obs_lbls_all).reshape(-1, 1)

    obs_data_tuple = (
        obs_token_ids.astype(int),
        obs_type_ids.astype(int),
        obs_position_ids.astype(int),
        obs_labels.astype(int),
    )

    data_dict = {'max_len={:d},eps={:.2f}'.format(data_config.max_len, data_config.epsilons[0]): obs_data_tuple}
    save_data(save_dir, data_dict, pretrain_mode='OBS_PRED')

    # process and save act data
    act_token_ids = np.concatenate(act_pred_all)
    act_type_ids, act_position_ids = compute_type_position_ids(act_token_ids, config)
    act_labels = np.array(act_lbls_all).reshape(-1, 1)

    act_data_tuple = (
        act_token_ids.astype(int),
        act_type_ids.astype(int),
        act_position_ids.astype(int),
        act_labels.astype(int),
    )

    data_dict = {'max_len={:d},eps={:.2f}'.format(data_config.max_len, data_config.epsilons[0]): act_data_tuple}
    save_data(save_dir, data_dict, pretrain_mode='ACT_PRED')

    print('Done!')

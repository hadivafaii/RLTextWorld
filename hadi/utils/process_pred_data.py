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

    parser.add_argument("game_type", help="(str) game type. (e.g. tw_cooking)", type=str)
    parser.add_argument("--game_spec", help="game specifics. default: None", type=str, default="",)

    args = parser.parse_args()

    import sys
    sys.path.append("..")
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
    _game_variants = ['train', 'valid', 'test']

    lang_dir_file = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(data_config.max_len))

    for gvar_id, game_var in enumerate(_game_variants):
        load_dir = pjoin(data_config.base_dirs[gvar_id], 'raw_pred_data')
        save_dir = data_config.pretrain_dirs[gvar_id]
        os.makedirs(save_dir, exist_ok=True)
        assert (game_var in load_dir) and (game_var in save_dir), "..."

        obs_pred_all = []
        obs_lbls_all = []
        act_pred_all = []
        act_lbls_all = []
        num_iters = len(os.listdir(load_dir))
        for i in range(num_iters):
            loaded_data_list = np.load(pjoin(load_dir, "iter={:d}.npy".format(i)), allow_pickle=True)
            for item in loaded_data_list:
                obs_pred_data, act_pred_data = item

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

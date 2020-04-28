import os
import sys
sys.path.append("..")

import numpy as np
import datetime
import h5py

from tqdm import tqdm
from itertools import permutations, chain

from model.configuration import TransformerConfig


# ---------------------------------------- Generate Permutation Data ------------------------------------ #
# ------------------------------------------------------------------------------------------------------- #
def _get_ranges(token_ids, config):
    _ids = [(np.where(x == config.obs_id)[0], np.where(x == config.act_id)[0]) for x in token_ids]
    obs_ids, act_ids = zip(*_ids)

    obs_ranges, act_ranges = [], []
    for i in range(len(token_ids)):
        oo, aa = obs_ids[i], act_ids[i]
        last_id = np.where(token_ids[i] > 0)[0][-1]
        aa = sorted(np.insert(aa, -1, last_id + 1))
        obs_ranges.append([range(tup[0], tup[1]) for tup in zip(oo, aa)])

        oo = np.delete(oo, 0)
        aa = np.delete(aa, -1)
        act_ranges.append([range(tup[0], tup[1]) for tup in zip(aa, oo)])
    return obs_ranges, act_ranges


def _permute_action_orders(arrs, gold_obs_ranges, gold_act_ranges, perm, mode='act'):
    if type(arrs) is not list:
        arrs = [arrs]

    if mode == 'act':
        assert len(gold_obs_ranges) - 1 == len(gold_act_ranges) == len(perm), 'there is len mismatch'
    elif mode == 'obs':
        assert len(gold_obs_ranges) == len(gold_act_ranges) + 1 == len(perm), 'there is len mismatch'
    else:
        raise NotImplementedError

    new_arrs = []
    for arr in arrs:
        permuted_arr = []
        if mode == 'act':
            for i, permuted_indx in enumerate(perm):
                permuted_arr.extend(arr[gold_obs_ranges[i]]) # add correct obs
                permuted_arr.extend(arr[gold_act_ranges[permuted_indx]]) # add permuted act
            permuted_arr.extend(arr[gold_obs_ranges[-1]]) # add the last obs
        elif mode == 'obs':
            for i, permuted_indx in enumerate(perm):
                permuted_arr.extend(arr[gold_obs_ranges[permuted_indx]]) # add correct obs
                if i < len(gold_act_ranges): # since always len(num_act) = len(num_obs) - 1
                    permuted_arr.extend(arr[gold_act_ranges[i]]) # add permuted act
        else:
            raise NotImplementedError

        permuted_arr = np.pad(permuted_arr, (0, len(arr) - len(permuted_arr))) # pad to correct length
        new_arrs.append(np.array(permuted_arr).astype(int))

    return new_arrs


def generate_permutated_data(inputs, config, k=3, mode='act'):
    token_ids, type_ides, position_ids = inputs

    new_token_ids = []
    new_type_ids = []
    new_position_ids = []
    permutations_used = []
    labels = []

    obs_ranges, act_ranges = _get_ranges(token_ids, config)

    for ii in tqdm(range(len(token_ids))):
        tokens = token_ids[ii]
        types = type_ids[ii]
        positions = position_ids[ii]
        oo = obs_ranges[ii]
        aa = act_ranges[ii]

        if mode == 'act':
            n = len(aa)
        elif mode == 'obs':
            n = len(oo)
        else:
            raise NotImplementedError

        # save the correct order one
        new_token_ids.append(tokens[np.newaxis, :])
        new_type_ids.append(types[np.newaxis, :])
        new_position_ids.append(positions[np.newaxis, :])
        permutations_used.append(list(range(n)))
        labels.extend([1])

        # save the permuted ones
        for jj in range(n - k + 1):
            non_identity_perms = list(permutations(range(jj, jj+k)))[1:] # perms[1:] : non identity
            non_identity_perms = [list(chain(range(jj), x, range(jj+k, n))) for x in non_identity_perms]

            for pp in non_identity_perms:
                permuted_token_ids, permuted_type_ids = _permute_action_orders([tokens, types], oo, aa, pp, mode)
                new_token_ids.append(permuted_token_ids[np.newaxis, :])
                new_type_ids.append(permuted_type_ids[np.newaxis, :])
                new_position_ids.append(positions[np.newaxis, :])
                permutations_used.append(pp)
                labels.extend([0])

    outputs = [new_token_ids, new_type_ids, new_position_ids]
    outputs = [np.concatenate(x, axis=0) for x in outputs]

    return tuple(outputs), np.expand_dims(labels, -1), permutations_used
# ------------------------------------------------------------------------------------------------------- #





# ------------------------------------------- Generate Corrupt Data ------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def detect_objects(x, ranges, conversion_dict):
    detected_ranges= []

    for range_ in ranges:
        unigram = x[range_]
        bigrams = list(zip(unigram, unigram[1:]))
        trigrams = list(zip(unigram, unigram[1:], unigram[2:]))
        quadrograms = list(zip(unigram, unigram[1:], unigram[2:], unigram[3:]))
        pentagrams = list(zip(unigram, unigram[1:], unigram[2:], unigram[3:], unigram[4:]))
        hexagrams = list(zip(unigram, unigram[1:], unigram[2:], unigram[3:], unigram[4:], unigram[5:]))

        for tuple_ in list(conversion_dict.keys()):
            detected = False
            if len(tuple_) == 1 and not detected:
                try:
                    start_idx = list(unigram).index(tuple_[0])
                    detected = True
                except ValueError:
                    continue

            elif len(tuple_) == 2:
                try:
                    start_idx = bigrams.index(tuple_)
                    detected = True
                except ValueError:
                    continue

            elif len(tuple_) == 3:
                try:
                    start_idx = trigrams.index(tuple_)
                    detected = True
                except ValueError:
                    continue

            elif len(tuple_) == 4:
                try:
                    start_idx = quadrograms.index(tuple_)
                    detected = True
                except ValueError:
                    continue

            elif len(tuple_) == 5:
                try:
                    start_idx = pentagrams.index(tuple_)
                    detected = True
                except ValueError:
                    continue

            elif len(tuple_) == 6:
                try:
                    start_idx = hexagrams.index(tuple_)
                    detected = True
                except ValueError:
                    continue
            else:
                raise ValueError('Max entity len should be 6')

            if detected:
                detected_range_ = range(list(range_)[start_idx], list(range_)[start_idx] + len(tuple_))
                detected_ranges.append((detected_range_, conversion_dict[tuple_]))

    return sorted(detected_ranges, key=lambda tup: tup[0].start)


def _get_masked_input(x, ranges, unk_id=3):
    extras = []
    for range_, _ in ranges:
        x = np.delete(x, range_)
        x = np.insert(x, range_.start, [unk_id] * len(range_))
        extras.append(range(range_.start + 1, range_.stop))
    extras = list(chain(*extras))
    return np.delete(x, extras)



def fix_inputs(x, starting_position_id=1, max_len=512):
    # pad token_ids so it gets to max
    x = np.pad(x, (0, max_len - len(x)), constant_values=0)

    # get the obs and act ranges for x
    obs_ranges, act_ranges = _get_ranges([x], config)
    obs_ranges, act_ranges = obs_ranges[0], act_ranges[0]

    # get the new type_ids
    type_ids = []
    for i in range(len(act_ranges)):
        type_ids.extend([config.obs_id] * len(obs_ranges[i]))
        type_ids.extend([config.act_id] * len(act_ranges[i]))
    type_ids.extend([config.obs_id] * (len(obs_ranges[-1])))
    type_ids = np.pad(type_ids, (0, max_len - len(type_ids)), constant_values=0)

    # get position_ids
    position_ids = np.arange(starting_position_id, starting_position_id + len(x[x > 0]))
    position_ids = np.pad(position_ids, (0, max_len - len(position_ids)), constant_values=0)

    return x[:max_len], type_ids[:max_len], position_ids[:max_len]


def generate_corrupted_data(inputs, config, conversion_dict, max_len=512, mask_prob=0.25, mode='act', seed=665):
    rng = np.random.RandomState(seed)
    token_ids, type_ides, position_ids = inputs

    assert token_ids.shape[-1] == type_ides.shape[-1] == position_ids.shape[-1] == max_len, "otherwise something is wrong"

    masked_token_ids = []
    masked_type_ids = []
    masked_position_ids = []
    unk_positions = []
    gold_labels = []

    obs_ranges, act_ranges = _get_ranges(token_ids, config)

    for ii in tqdm(range(len(token_ids))):
        # get ranges for objects of interest
        if mode == 'act':
            detected_ranges = detect_objects(token_ids[ii], act_ranges[ii], conversion_dict)
        elif mode == 'obs':
            detected_ranges = detect_objects(token_ids[ii], obs_ranges[ii], conversion_dict)
        elif mode == 'mlm':
            detected_ranges = [(range(j, j + 1), token_ids[ii][j]) for j in range(np.sum(token_ids[ii] != config.pad_id))]
        else:
            raise NotImplementedError

        # randomly select form detected objects
        num_ = len(detected_ranges)
        random_numbers = rng.uniform(0, 1, num_)
        m = sorted(np.where(random_numbers < mask_prob)[0])
        # mask the selected objects
        x_masked = _get_masked_input(token_ids[ii], [detected_ranges[x] for x in m], config.unk_id)
        # fix the masked inputs to have correct length, position and type ids etc
        outputs_ = fix_inputs(x_masked, position_ids[ii][0], max_len=max_len)

        masked_token_ids.append(outputs_[0])
        masked_type_ids.append(outputs_[1])
        masked_position_ids.append(outputs_[2])

        unk_positions.append(np.where(outputs_[0] == config.unk_id)[0])
        gold_labels.append([[tup[1] for tup in detected_ranges][x] for x in m])

    outputs = (np.array(masked_token_ids), np.array(masked_type_ids), np.array(masked_position_ids))
    return outputs, unk_positions, gold_labels
# ------------------------------------------------------------------------------------------------------- #





# ----------------------------------------------- Save & Load ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def save_data(save_dir, token_ids, type_ids, position_ids,
              labels=None, permutations_used=None, unk_positions=None,
              pretrain_mode='ACT_ORDER', max_len=512, eps=1.00):
    save_ = os.path.join(save_dir, "{:s}_{:s}.hdf5".format(pretrain_mode, datetime.datetime.now().strftime("[%Y_%m_%d_%H:%M]")))
    f = h5py.File(save_, "w")
    f.create_group(pretrain_mode).create_group('max_len={:d}'.format(max_len)).create_group('eps={:.2f}'.format(eps))
    subgroup = f[pretrain_mode]['max_len={:d}'.format(max_len)]['eps={:.2f}'.format(eps)]

    subgroup.create_dataset('token_ids', data=token_ids)
    subgroup.create_dataset('type_ids', data=type_ids)
    subgroup.create_dataset('position_ids', data=position_ids)

    if labels is not None:
        if type(labels) == list:
            dset = subgroup.create_dataset('labels', (len(labels),),
                                           dtype=h5py.vlen_dtype(np.dtype('int32')))
            for i in range(len(labels)):
                dset[i] = labels[i]
        else:
            subgroup.create_dataset('labels', data=np.array(labels))

    if permutations_used is not None:
        dset = subgroup.create_dataset('permutations_used', (len(permutations_used),),
                                       dtype=h5py.vlen_dtype(np.dtype('int32')))
        for i in range(len(permutations_used)):
            dset[i] = permutations_used[i]

    if unk_positions is not None:
        dset = subgroup.create_dataset('unk_positions', (len(unk_positions),),
                                       dtype=h5py.vlen_dtype(np.dtype('int32')))
        for i in range(len(unk_positions)):
            dset[i] = unk_positions[i]

    print('Data saved at {:s}'.format(save_))
    f.close()


def load_data(game_type, pretrain_mode, game_specs='', mask_prob=None, k=None, max_len=512, eps=1.00, file_name=None):

    base_dir = os.path.join(os.environ['HOME'], 'Documents/FTWP/trajectories', game_type, game_specs, 'pretraining_data')
    if pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB'] and mask_prob is None:
        dir_list = sorted([x for x in os.listdir(base_dir) if 'mask_prob=' in x])
    elif pretrain_mode in ['ACT_ORDER', 'OBS_ORDER'] and k is None:
        dir_list = sorted([x for x in os.listdir(base_dir) if 'k=' in x])
    else:
        raise ValueError("Wrong pretrain type entered.  Allowed options:\n{}".format(ALLOWED_MODES))

    assert len(dir_list) > 0, 'no dirs found for pretrain type {:s}'.format(pretrain_mode)
    print('Found these dirs:\n', dir_list)
    load_dir = os.path.join(base_dir, dir_list[-1])

    if file_name is None:
        file_list = os.listdir(load_dir)
        file_list = sorted([x for x in file_list if '{:s}'.format(pretrain_mode) in x])
        assert len(file_list) > 0, 'no files found for pretrain type {:s}'.format(pretrain_mode)
        print('Found these fils:\n', file_list)
        file_name = file_list[-1]

    load_ = os.path.join(load_dir, file_name)
    print('\nLoading data from {:s}'.format(load_))

    f = h5py.File(load_, "r")
    subgroup = f[pretrain_mode]['max_len={:d}'.format(max_len)]['eps={:.2f}'.format(eps)]

    token_ids = np.array(subgroup['token_ids'])
    type_ids = np.array(subgroup['type_ids'])
    position_ids = np.array(subgroup['position_ids'])

    outputs = (token_ids, type_ids, position_ids)

    if 'labels' in subgroup.keys():
        if subgroup['labels'].dtype == 'object':
            dset = subgroup['labels']
            labels = []
            for pp in dset:
                labels.append(list(pp))
        else:
            labels = np.array(subgroup['labels'])
        outputs += (labels,)
        print('labels added to outputs')

    if 'permutations_used' in subgroup.keys():
        dset = subgroup['permutations_used']
        permutations_used = []
        for pp in dset:
            permutations_used.append(list(pp))
        outputs += (permutations_used,)
        print('permutations_used added to outputs')

    if 'unk_positions' in subgroup.keys():
        dset = subgroup['unk_positions']
        unk_positions = []
        for pp in dset:
            unk_positions.append(list(pp))
        outputs += (unk_positions,)
        print('unk_positions added to outputs')

    f.close()
    return outputs
# ------------------------------------------------------------------------------------------------------- #




# ------------------------------------------ Helper Functions ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def _corrupted_data_processing(generated_data_):
    curropt_token_ids = [tup[0][0] for tup in generated_data_]
    curropt_type_ids = [tup[0][1] for tup in generated_data_]
    curropt_position_ids = [tup[0][2] for tup in generated_data_]
    outputs = (curropt_token_ids, curropt_type_ids, curropt_position_ids)
    outputs = (np.concatenate(x, axis=0) for x in outputs)

    unk_positions_and_labels_list = [tup[1:] for tup in generated_data_]
    unk_positions, gold_labels = [], []
    for item1, item2 in unk_positions_and_labels_list:
        unk_positions += item1
        gold_labels += item2

    return outputs, unk_positions, gold_labels
# ------------------------------------------------------------------------------------------------------- #




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("game_type", help="(str) game type. (e.g. tw_cooking/train", type=str)
    parser.add_argument("pretrain_mode", help="(str) pretrain data type", type=str)
    parser.add_argument("--k", help="(integer >= 2) will have k! permutations.  defautl: 3", type=int, default=3)
    parser.add_argument("--mask_prob", help="(float) portion of data to distort.  defautl: 0.30", type=float, default=0.30)
    parser.add_argument("--S", help="(integer) sequence length. default: 512", type=int, default=512)
    parser.add_argument("--eps", help="(float) epsilon. default: 1.00", type=float, default=1.00)
    parser.add_argument("--seeds", help="(integers) random seeds, used only for corrupted data generation. default: [665]", type=int, nargs='+', default=665)
    parser.add_argument("--game_specs", help="game specifics such as brief or detailed goal, quest length and so on. default is None",
                        type=str, default="")
    parser.add_argument("--save_dir", help="(str) directory to load and save. default: '~/game_type/pretraining_data'",
                        type=str, default='pretraining_data')

    args = parser.parse_args()

    ALLOWED_MODES = [
        'ACT_ORDER', 'ACT_ENTITY', 'ACT_VERB',
        'OBS_ORDER', 'OBS_ENTITY', 'OBS_VERB', 'MLM']

    if args.pretrain_mode not in ALLOWED_MODES:
        raise ValueError('enter correct pretrain type.  allowed opetions: \n{}'.format(ALLOWED_MODES))

    base_dir = os.path.join('/home/hadivafa/Documents/FTWP/trajectories', args.game_type, args.game_specs)

    load_dir = os.path.join(base_dir, 'processed_trajectories')
    save_dir = os.path.join(base_dir, args.save_dir)

    traj_load_ = os.path.join(load_dir, 'traj_data_max_len={:d}.npy'.format(args.S))
    lang_load_ = os.path.join(load_dir, 'lang_data_max_len={:d}.npy'.format(args.S))
    traj_data_all = np.load(traj_load_, allow_pickle=True).item()
    lang_data_all = np.load(lang_load_, allow_pickle=True).item()
    traj_data = traj_data_all['eps={:.2f}'.format(args.eps)]
    lang_data = lang_data_all['eps={:.2f}'.format(args.eps)]

    token_ids = traj_data['sequence_ids']
    type_ids = traj_data['type_ids']
    segment_ids = traj_data['segment_ids']
    position_ids = traj_data['position_ids']
    masks = traj_data['masks']

    inputs = (token_ids, type_ids, position_ids)
    config = TransformerConfig()

    generated_data_ = []

    if args.pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
        outputs, labels, permutations_used = generate_permutated_data(
            inputs, config, k=args.k, mode=args.pretrain_mode[:3].lower(),
        )
        save_dir = os.path.join(save_dir, "k={:d}".format(args.k))
        os.makedirs(save_dir, exist_ok=True)
        save_data(save_dir, *outputs, labels=labels, permutations_used=permutations_used,
                  pretrain_mode=args.pretrain_mode, max_len=args.S, eps=args.eps)

    elif args.pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB']:
        for seed in args.seeds:
            print('[PROGRESS] generating corrupted data using seed = {:d}'.format(seed))
            outputs, unk_positions, gold_labels = generate_corrupted_data(
                inputs, config, lang_data['{:s}2indx'.format(args.pretrain_mode[4:].lower())],
                max_len=args.S, mask_prob=args.mask_prob, mode=args.pretrain_mode[:3].lower(), seed=seed,
            )
            generated_data_.append((outputs, unk_positions, gold_labels))
        outputs, unk_positions, gold_labels = _corrupted_data_processing(generated_data_)
        print('[PROGRESS] data generation done. saving . . . ')
        save_dir = os.path.join(save_dir, "mask_prob={:.2f}".format(args.mask_prob))
        os.makedirs(save_dir, exist_ok=True)
        save_data(save_dir, *outputs, unk_positions=unk_positions, labels=gold_labels,
                  pretrain_mode=args.pretrain_mode, max_len=args.S, eps=args.eps)
    else:
        raise NotImplementedError

    print('Done!')

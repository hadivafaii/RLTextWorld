import os
import sys
import numpy as np
import datetime
import h5py
from tqdm import tqdm
from itertools import permutations, chain


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
                permuted_arr.extend(arr[gold_obs_ranges[i]])  # add correct obs
                permuted_arr.extend(arr[gold_act_ranges[permuted_indx]])  # add permuted act
            permuted_arr.extend(arr[gold_obs_ranges[-1]])  # add the last obs
        elif mode == 'obs':
            for i, permuted_indx in enumerate(perm):
                permuted_arr.extend(arr[gold_obs_ranges[permuted_indx]])  # add correct obs
                if i < len(gold_act_ranges):  # since always len(num_act) = len(num_obs) - 1
                    permuted_arr.extend(arr[gold_act_ranges[i]])  # add permuted act
        else:
            raise NotImplementedError

        permuted_arr = np.pad(permuted_arr, (0, len(arr) - len(permuted_arr)))  # pad to correct length
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
            non_identity_perms = list(permutations(range(jj, jj + k)))[1:]  # perms[1:] : non identity
            non_identity_perms = [list(chain(range(jj), x, range(jj + k, n))) for x in non_identity_perms]

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
    detected_ranges = []

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
    labels = []

    obs_ranges, act_ranges = _get_ranges(token_ids, config)

    for ii in tqdm(range(len(token_ids))):
        # get ranges for objects of interest
        if mode == 'act':
            detected_ranges = detect_objects(token_ids[ii], act_ranges[ii], conversion_dict)
        elif mode == 'obs':
            detected_ranges = detect_objects(token_ids[ii], obs_ranges[ii], conversion_dict)
        elif mode == 'mlm':
            detected_ranges = [(range(j, j + 1), token_ids[ii][j]) for j in
                               range(int(np.sum(token_ids[ii] != config.pad_id)))]
        else:
            raise NotImplementedError

        # randomly select form detected objects
        num_ = len(detected_ranges)
        if num_ == 0:
            continue
        m = []
        while len(m) == 0:  # this will ensure at least one masked token
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
        labels.append([[tup[1] for tup in detected_ranges][x] for x in m])

    outputs = [np.array(masked_token_ids), np.array(masked_type_ids), np.array(masked_position_ids)]
    return outputs, labels, unk_positions


# ------------------------------------------------------------------------------------------------------- #


# ----------------------------------------------- Save & Load ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def save_data(save_dir, data_dict, pretrain_mode='ACT_ORDER'):
    save_ = os.path.join(save_dir, "{:s}_{:s}.hdf5".format(
        pretrain_mode, datetime.datetime.now().strftime("[%Y_%m_%d_%H:%M]")))
    f = h5py.File(save_, "w")
    pretrain_group = f.create_group(pretrain_mode)

    for key, data_ in data_dict.items():
        xtract = list(map(lambda x: x.split('='), key.split(',')))
        max_len, eps = int(xtract[0][1]), float(xtract[1][1])

        try:
            pretrain_group.create_group('max_len={:d}'.format(max_len)).create_group('eps={:.2f}'.format(eps))
        except ValueError:
            pretrain_group['max_len={:d}'.format(max_len)].create_group('eps={:.2f}'.format(eps))

        subgroup = pretrain_group['max_len={:d}'.format(max_len)]['eps={:.2f}'.format(eps)]

        subgroup.create_dataset('token_ids', data=data_[0])
        subgroup.create_dataset('type_ids', data=data_[1])
        subgroup.create_dataset('position_ids', data=data_[2])

        labels = data_[3]
        if type(labels) is list:
            dset = subgroup.create_dataset('labels', (len(labels),),
                                           dtype=h5py.vlen_dtype(np.dtype('int32')))
            for i in range(len(labels)):
                dset[i] = labels[i]
        else:
            subgroup.create_dataset('labels', data=np.array(labels))

        # this one is unk_positions for CORRUPT data and permutation_used for PERMUTED data
        other_data = data_[4]
        dset = subgroup.create_dataset('other_data', (len(other_data),),
                                       dtype=h5py.vlen_dtype(np.dtype('int32')))
        for i in range(len(other_data)):
            dset[i] = other_data[i]

    print('Data saved at {:s}'.format(save_))
    f.close()


def load_data(data_config, file_name=None):
    load_data = []
    for pretrain_mode, load_dir in zip(data_config.pretrain_modes, data_config.pretrain_dirs):
        if file_name is None:
            file_list = os.listdir(load_dir)
            file_list = sorted([x for x in file_list if '{:s}'.format(pretrain_mode) in x])
            assert len(file_list) > 0, 'no files found for pretrain type {:s}'.format(pretrain_mode)
            print('Found these files:\n', file_list)
            file_name = file_list[-1]

        load_ = os.path.join(load_dir, file_name)
        print('\nLoading data from {:s}\n'.format(load_))

        f = h5py.File(load_, "r")
        pretrain_group = f[pretrain_mode]

        data_dict = {}
        for max_len_key in tqdm(pretrain_group):
            for eps_key in pretrain_group[max_len_key]:
                subgroup = pretrain_group[max_len_key][eps_key]

                token_ids = np.array(subgroup['token_ids'])
                type_ids = np.array(subgroup['type_ids'])
                position_ids = np.array(subgroup['position_ids'])

                outputs = (token_ids, type_ids, position_ids)

                if subgroup['labels'].dtype == 'object':
                    dset = subgroup['labels']
                    labels = []
                    for pp in dset:
                        labels.append(list(pp))
                else:
                    labels = np.array(subgroup['labels'])
                outputs += (labels,)

                dset = subgroup['other_data']
                other_data = []
                for pp in dset:
                    other_data.append(list(pp))
                outputs += (other_data,)

                max_len, eps = int(max_len_key.split('=')[1]), float(eps_key.split('=')[1])
                key = 'max_len={:d},eps={:.2f}'.format(max_len, eps)
                data_dict.update({key: outputs})
        load_data.append(data_dict)
        f.close()
    return load_data


# ------------------------------------------------------------------------------------------------------- #


# ------------------------------------------ Helper Functions ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def _corrupted_data_processing(generated_data_):
    curropt_token_ids = [ll[0] for ll in generated_data_]
    curropt_type_ids = [ll[1] for ll in generated_data_]
    curropt_position_ids = [ll[2] for ll in generated_data_]
    outputs = [curropt_token_ids, curropt_type_ids, curropt_position_ids]
    outputs = [np.concatenate(x, axis=0) for x in outputs]

    labels_unk_positions_list = [ll[-2:] for ll in generated_data_]
    labels, unk_positions = [], []
    for item1, item2 in labels_unk_positions_list:
        labels += item1
        unk_positions += item2

    return outputs, labels, unk_positions


# ------------------------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("game_type", help="(str) game type. (e.g. tw_cooking/train", type=str)
    parser.add_argument("pretrain_mode", help="(str) pretrain data type", type=str)
    parser.add_argument("--k", help="(integer >= 2) will have k! permutations.  defautl: 3", type=int, default=3)
    parser.add_argument("--mask_prob", help="(float) portion of data to distort.  defautl: 0.30", type=float,
                        default=0.30)
    parser.add_argument("--max_lengths", help="(integer) sequence lengths. default: [384, 512, 768, 1024, 2048]",
                        type=int, default=None)
    parser.add_argument("--eps_step",
                        help="(float) epsilon increment steps, i.e. eps in np.arange(0.0, eps_step * 11, eps_step). default: 0.10",
                        type=float, default=0.10)
    parser.add_argument("--seeds",
                        help="(integers) random seeds, used only for corrupted data generation. default: [665]",
                        type=int, nargs='+', default=665)
    parser.add_argument("--game_spec",
                        help="game specifics such as brief or detailed goal, quest length and so on. default is None",
                        type=str, default="")

    args = parser.parse_args()

    ALLOWED_MODES = [
        'ACT_ORDER', 'ACT_ENTITY', 'ACT_VERB',
        'OBS_ORDER', 'OBS_ENTITY', 'OBS_VERB', 'MLM']

    if args.pretrain_mode not in ALLOWED_MODES:
        raise ValueError('enter correct pretrain type.  allowed opetions: \n{}'.format(ALLOWED_MODES))

    assert args.k >= 2, "k should be integer greater than 1"
    assert 0 < args.mask_prob < 1, "mask_prob should be in (0, 1) interval"

    sys.path.append("..")
    from model.preprocessing import get_nlp, preproc
    from model.configuration import DataConfig, TransformerConfig

    config = TransformerConfig()
    data_config = DataConfig(pretrain_modes=args.pretrain_mode, game_type=args.game_type,
                             game_spec=args.game_spec, k=args.k, mask_prob=args.mask_prob)

    load_dir = data_config.processed_dir
    save_dir = data_config.pretrain_dirs[0]

    if args.max_lengths is None:
        max_lengths = [384, 512, 768, 1024, 2048]
    else:
        max_lengths = [args.max_lengths]

    # generate data loop
    loop_data = {}
    for max_len in max_lengths:
        for eps in np.arange(0.0, args.eps_step * 11, args.eps_step):
            print('\n\n')
            print('-' * 25, 'max_len={:d},eps={:.2f}'.format(max_len, eps), '-' * 25)
            traj_load_ = os.path.join(load_dir, 'traj_data_max_len={:d}.npy'.format(max_len))
            lang_load_ = os.path.join(load_dir, 'lang_data_max_len={:d}.npy'.format(max_len))
            traj_data_all = np.load(traj_load_, allow_pickle=True).item()
            lang_data_all = np.load(lang_load_, allow_pickle=True).item()
            traj_data = traj_data_all['eps={:.2f}'.format(eps)]
            lang_data = lang_data_all['eps={:.2f}'.format(eps)]

            token_ids = traj_data['sequence_ids']
            type_ids = traj_data['type_ids']
            position_ids = traj_data['position_ids']

            inputs = (token_ids, type_ids, position_ids)

            if args.pretrain_mode in ['ACT_ORDER', 'OBS_ORDER']:
                outputs, labels, permutations_used = generate_permutated_data(
                    inputs, config, k=args.k, mode=args.pretrain_mode[:3].lower(),
                )
                loop_data.update(
                    {'max_len={:d},eps={:.2f}'.format(max_len, eps): (*outputs, labels, permutations_used)})

            elif args.pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB']:
                generated_data_ = []
                for seed in args.seeds:
                    print('[PROGRESS] generating corrupted data using seed = {:d}'.format(seed))
                    outputs, labels, unk_positions = generate_corrupted_data(
                        inputs, config, lang_data['{:s}2indx'.format(args.pretrain_mode[4:].lower())],
                        max_len=max_len, mask_prob=args.mask_prob, mode=args.pretrain_mode[:3].lower(), seed=seed,
                    )
                    generated_data_.append([*outputs, labels, unk_positions])
                outputs, labels, unk_positions = _corrupted_data_processing(generated_data_)
                loop_data.update({'max_len={:d},eps={:.2f}'.format(max_len, eps): (*outputs, labels, unk_positions)})

            else:
                raise NotImplementedError

    # save
    os.makedirs(save_dir, exist_ok=True)
    save_data(save_dir, loop_data, args.pretrain_mode)

    print('Done!')

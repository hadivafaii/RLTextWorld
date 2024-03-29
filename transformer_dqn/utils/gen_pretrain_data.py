import os
from os.path import join as pjoin
import sys
import torch
import numpy as np
import datetime
import h5py
from copy import deepcopy as dc
from tqdm import tqdm
from itertools import permutations, chain


# ---------------------------------------- Generate Permutation Data ------------------------------------ #
# ------------------------------------------------------------------------------------------------------- #
def get_ranges(token_ids, config, flatten=False):
    where_fn = np.where
    if type(token_ids) == torch.Tensor:
        where_fn = torch.where

    _ids = [(where_fn(x == config.obs_id)[0], where_fn(x == config.act_id)[0]) for x in token_ids]
    obs_ids, act_ids = zip(*_ids)

    # takes [TRAJ] into account
    obs_ids = [where_fn(item == 1, 0, item) for item in obs_ids]

    max_len = len(token_ids[0])

    obs_ranges, act_ranges = [], []
    for i in range(len(token_ids)):
        oo, aa = obs_ids[i], act_ids[i]
        last_id = where_fn(token_ids[i] > 0)[0][-1]
        aa = sorted(np.insert(aa, -1, last_id + 1))
        if flatten:
            obs_ranges.extend([range(tup[0] + (i * max_len), tup[1] + (i * max_len)) for tup in zip(oo, aa)])
        else:
            obs_ranges.append([range(tup[0], tup[1]) for tup in zip(oo, aa)])

        oo = np.delete(oo, 0)
        aa = np.delete(aa, -1)
        if flatten:
            act_ranges.extend([range(tup[0] + (i * max_len), tup[1] + (i * max_len)) for tup in zip(aa, oo)])
        else:
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

    obs_ranges, act_ranges = get_ranges(token_ids, config)

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


def _needs_fixing(x):
    x_needs_fixing = False

    for indx, ((range1, _), (range2, _)) in enumerate(zip(x, x[1:])):
        if range1.stop > range2.start:
            x_needs_fixing = True

    return x_needs_fixing


def _fix_ranges(detected_ranges):
    x = detected_ranges.copy()

    del_indx = None
    for indx, ((range1, _), (range2, _)) in enumerate(zip(x, x[1:])):
        if range1.stop > range2.start:
            if len(range1) >= len(range2):
                del_indx = indx + 1
            elif len(range1) < len(range2):
                del_indx = indx
    if del_indx is not None:
        del x[del_indx]
    return x


def fix_detected_ranges(detected_ranges):
    x = sorted(detected_ranges, key=lambda tup: tup[0].start).copy()
    while _needs_fixing(x):
        x = _fix_ranges(x)
    return x


def _get_masked_input(x, ranges, unk_id=3):
    extras = []
    for range_, _ in ranges:
        x = np.delete(x, range_)
        x = np.insert(x, range_.start, [unk_id] * len(range_))
        extras.append(range(range_.start + 1, range_.stop))
    extras = list(chain(*extras))
    return np.delete(x, extras)


def compute_type_position_ids(x, config, starting_position_ids=None):
    if starting_position_ids is None:
        starting_position_ids = np.ones(len(x))

    position_ids = np.concatenate(
        [np.expand_dims(np.pad(np.arange(
            tup[0], tup[0] + tup[1]),
            (0, x.shape[-1] - tup[1]), constant_values=config.pad_id), axis=0)
            for tup in zip(starting_position_ids, (x > 0).sum(1))]
    )

    # The [TRAJ] always has position = 1
    # position_ids[:, 0] = np.ones(len(x))

    type_ids = (np.ones(x.shape, dtype=int) * config.pad_id).flatten()

    obs_ranges_flat, act_ranges_flat = get_ranges(x, config, flatten=True)
    act_ranges_flat_chained = list(chain(*act_ranges_flat))
    obs_ranges_flat_chained = list(chain(*obs_ranges_flat))

    type_ids[obs_ranges_flat_chained] = 1
    type_ids[act_ranges_flat_chained] = 2

    return type_ids.reshape(x.shape).astype(int), position_ids.astype(int)


def generate_corrupted_data(inputs, config, conversion_dict=None, bag_of_object_tokens=None,
                            reset_positions=True, max_len=512, mask_prob=0.25, mode='MLM', seed=665):
    # TODO: if you actualy verified that ACT_ENTITY etc are useless, come back and fix this
    #  function to make it faster. Remove the whole detect ranges thing and mask the flattened input
    rng = np.random.RandomState(seed)
    token_ids, type_ides, position_ids = inputs

    assert token_ids.shape[-1] == type_ides.shape[-1] == position_ids.shape[-1] == max_len, "otherwise something is wrong"

    masked_token_ids = np.ones(token_ids.shape, dtype=int) * config.pad_id
    starting_pos_ids = np.ones(len(token_ids), dtype=int)
    labels_arr = np.ones(token_ids.shape) * -100

    obs_ranges, act_ranges = get_ranges(token_ids, config)

    for ii in tqdm(range(len(token_ids)), desc='mask_prob: {:.2f}, mdoe: {:s}'.format(mask_prob, mode)):
        # get ranges for objects of interest
        if 'ACT' in mode:
            detected_ranges = detect_objects(token_ids[ii], act_ranges[ii], conversion_dict)
        elif 'OBS' in mode:
            detected_ranges = detect_objects(token_ids[ii], obs_ranges[ii], conversion_dict)
        elif mode == 'MLM':
            detected_ranges = [(range(j, j + 1), token_ids[ii][j]) for j in
                               range(int(np.sum(token_ids[ii] != config.pad_id)))
                               if token_ids[ii][j] not in [config.pad_id, config.obs_id, config.act_id, config.traj_id]]
        elif mode == 'MOM':
            detected_ranges = [(range(j, j + 1), token_ids[ii][j]) for j in
                               range(int(np.sum(token_ids[ii] != config.pad_id)))
                               if token_ids[ii][j] in bag_of_object_tokens]
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
        # fix the selected detected ranges to avoid overlab
        detected_ranges_fixed = fix_detected_ranges([detected_ranges[x] for x in m])
        if len(detected_ranges_fixed) == 0:
            continue
        # mask the selected objects
        x_masked = _get_masked_input(token_ids[ii], detected_ranges_fixed, config.unk_id)
        masked_token_ids[ii] = np.pad(x_masked[:max_len], (0, max_len - len(x_masked)), constant_values=config.pad_id)
        labels_arr[ii][np.where(masked_token_ids[ii] == config.unk_id)[0]] = [tup[1] for tup in detected_ranges_fixed]

    # remove empty rows due to occurances of num_ = 0
    masked_token_ids = masked_token_ids[~np.all(masked_token_ids == config.pad_id, axis=1)]
    labels_arr = labels_arr[~np.all(labels_arr == -100, axis=1)]
    masked_type_ids, masked_position_ids = compute_type_position_ids(masked_token_ids, config, starting_pos_ids)

    if reset_positions:
        outputs = ([masked_token_ids, masked_type_ids, masked_position_ids], labels_arr.astype(int))
    else:
        outputs = ([masked_token_ids, masked_type_ids, position_ids], labels_arr.astype(int))

    return outputs


# ------------------------------------------------------------------------------------------------------- #


# ----------------------------------------------- Save & Load ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def save_data(save_dir, data_dict, pretrain_mode='ACT_ORDER'):
    save_ = pjoin(save_dir, "{:s}_{:s}.hdf5".format(
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

        subgroup.create_dataset('token_ids', data=data_[0], dtype=int)
        subgroup.create_dataset('type_ids', data=data_[1], dtype=int)
        subgroup.create_dataset('position_ids', data=data_[2], dtype=int)
        subgroup.create_dataset('labels', data=data_[3], dtype=int)

        # this one is is permutations used for PERMUTE data
        for i in range(4, len(data_)):
            other_data = data_[i]
            dset = subgroup.create_dataset('other_data_{:d}'.format(i-4), (len(other_data),),
                                           dtype=h5py.vlen_dtype(np.dtype('int32')))
            for j in range(len(other_data)):
                dset[j] = other_data[j]

    print('Data saved at {:s}'.format(save_))
    f.close()


def load_data(data_config, file_names=None, load_extra_stuff=False, verbose=False):
    if data_config.train_valid_test:
        ratio = 3
    else:
        ratio = 1

    if file_names is None:
        file_names = [None] * len(data_config.pretrain_dirs)
    else:
        assert len(file_names) == len(data_config.pretrain_dirs), "must provide a file name for each load_dir"

    load_data = {}
    loaded_from = [None] * len(data_config.pretrain_dirs)
    for i, pretrain_mode in enumerate(data_config.pretrain_modes):
        for j in range(ratio*i, ratio*(i+1)):
            load_dir = data_config.pretrain_dirs[j]
            game_type = data_config.game_types[j % ratio]

            if file_names[j] is None:
                try:
                    file_list = os.listdir(load_dir)
                    file_list = sorted([x for x in file_list if '{:s}'.format(pretrain_mode) in x])
                except FileNotFoundError:
                    continue
                if len(file_list) == 0:
                    print('No files found for pretrain type: \n {:s} \n At: \n {:s}'.format(pretrain_mode, load_dir))
                    continue
                else:
                    if verbose:
                        print('Found these files:\n', file_list)
                    file_name = file_list[-1]
            else:
                file_name = file_names[j]

            load_ = pjoin(load_dir, file_name)
            loaded_from[j] = load_
            if verbose:
                print('\nLoading data from {:s}\n'.format(load_))

            with h5py.File(load_, "r") as f:
                pretrain_group = f[pretrain_mode]

                data_dict = {}
                for max_len_key in pretrain_group:
                    for eps_key in pretrain_group[max_len_key]:
                        max_len, eps = int(max_len_key.split('=')[1]), float(eps_key.split('=')[1])
                        if not (max_len == data_config.max_len and eps in data_config.epsilons):
                            continue

                        subgroup = pretrain_group[max_len_key][eps_key]

                        token_ids = np.array(subgroup['token_ids'])
                        type_ids = np.array(subgroup['type_ids'])
                        position_ids = np.array(subgroup['position_ids'])
                        labels = np.array(subgroup['labels'])

                        outputs = (token_ids, type_ids, position_ids, labels)

                        if load_extra_stuff:
                            for k in range(4, len(subgroup)):
                                dset = subgroup['other_data_{:d}'.format(k-4)]
                                other_data = []
                                for pp in dset:
                                    other_data.append(list(pp))
                                outputs += (other_data,)

                        key = 'max_len={:d},eps={:.2f}'.format(max_len, eps)
                        data_dict.update({key: outputs})
                type_key = "{:s}-{:s}".format(pretrain_mode, game_type)
                load_data.update({type_key: data_dict})
    return load_data, loaded_from


# ------------------------------------------------------------------------------------------------------- #


# ------------------------------------------ Helper Functions ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def _corrupted_data_processing(generated_data_):
    curropt_token_ids = [ll[0] for ll in generated_data_]
    curropt_type_ids = [ll[1] for ll in generated_data_]
    curropt_position_ids = [ll[2] for ll in generated_data_]
    outputs = [curropt_token_ids, curropt_type_ids, curropt_position_ids]
    outputs = [np.concatenate(x, axis=0) for x in outputs]

    labels = np.concatenate([ll[3] for ll in generated_data_], axis=0)

    return outputs, labels


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
                        type=float, default=1.0)
    parser.add_argument("--seeds",
                        help="(integers) random seeds, used only for corrupted data generation. default: [665]",
                        type=int, nargs='+', default=665)
    parser.add_argument("--game_spec",
                        help="game specifics such as brief or detailed goal, quest length and so on. default is None",
                        type=str, default="")

    args = parser.parse_args()

    ALLOWED_MODES = [
        'MLM', 'MOM',
        'ACT_ENTITY', 'ACT_VERB',
        'OBS_ENTITY', 'OBS_VERB',
        'ACT_ORDER', 'OBS_ORDER',
        'ACT_PREDICT', 'OBS_PREDICT', 'ACT_ELIM',
    ]

    if args.pretrain_mode not in ALLOWED_MODES:
        raise ValueError('enter correct pretrain type.  allowed opetions: \n{}'.format(ALLOWED_MODES))

    assert args.k >= 2, "k should be integer greater than 1"
    assert 0 < args.mask_prob < 1, "mask_prob should be in (0, 1) interval"

    sys.path.append("..")
    from model.configuration import DataConfig, TransformerConfig

    config = TransformerConfig()
    data_config = DataConfig(pretrain_modes=args.pretrain_mode, game_type=args.game_type,
                             game_spec=args.game_spec, k=args.k, train_valid_test=True,
                             mlm_mask_prob=args.mask_prob, mom_mask_prob=args.mask_prob)

    if args.game_type.split('/')[1] == 'train':
        load_dir = data_config.processed_dirs[0]
        save_dir = data_config.pretrain_dirs[0]
        assert 'train' in load_dir and 'train' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'valid':
        load_dir = data_config.processed_dirs[1]
        save_dir = data_config.pretrain_dirs[1]
        assert 'valid' in load_dir and 'valid' in save_dir, "..."
    elif args.game_type.split('/')[1] == 'test':
        load_dir = data_config.processed_dirs[2]
        save_dir = data_config.pretrain_dirs[2]
        assert 'test' in load_dir and 'test' in save_dir, "..."
    else:
        raise ValueError("Invalid game type entered (e.g. custom/train is correct)")

    if args.max_lengths is None:
        max_lengths = [384, 512, 768, 1024]
    else:
        max_lengths = [args.max_lengths]

    # generate data loop
    loop_data = {}
    for max_len in max_lengths:
        for eps in np.arange(0.0, args.eps_step + 1, args.eps_step):
            if not (eps == 1.00 and max_len in [512]):     # not necessary for now
                continue
            print('\n\n')
            print('-' * 25, 'max_len={:d},eps={:.2f}'.format(max_len, eps), '-' * 25)
            traj_load_ = pjoin(load_dir, 'traj_data_max_len={:d}.npy'.format(max_len))
            lang_load_ = pjoin(data_config.lang_dir, 'lang_data_max_len={:d}.npy'.format(max_len))
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
                    inputs=inputs,
                    config=config,
                    k=args.k,
                    mode=args.pretrain_mode[:3].lower(),
                )
                loop_data.update(
                    {'max_len={:d},eps={:.2f}'.format(max_len, eps): (*outputs, labels, permutations_used)})

            elif args.pretrain_mode in ['ACT_ENTITY', 'ACT_VERB', 'OBS_ENTITY', 'OBS_VERB', 'MLM', 'MOM']:
                generated_data_ = []
                for seed in args.seeds:
                    print('[PROGRESS] generating corrupted data using seed = {:d}'.format(seed))
                    if args.pretrain_mode == 'MLM':
                        conversion_dict = None
                        bag_of_object_tokens = None

                    elif args.pretrain_mode == 'MOM':
                        conversion_dict = None
                        entities = lang_data['entity2indx'].keys()
                        verbs = lang_data['verb2indx'].keys()

                        bag_of_object_tokens = []
                        for verb_tup in verbs:
                            for tok_id in verb_tup:
                                bag_of_object_tokens.append(tok_id)
                        for ent_tup in entities:
                            for tok_id in ent_tup:
                                bag_of_object_tokens.append(tok_id)
                        bag_of_object_tokens = np.unique(bag_of_object_tokens)
                        # TODO: delete "of" from here

                    else:
                        conversion_dict = lang_data['{:s}2indx'.format(args.pretrain_mode[4:].lower())]
                        bag_of_object_tokens = None

                    outputs, labels = generate_corrupted_data(
                        inputs=inputs,
                        config=config,
                        conversion_dict=conversion_dict,
                        bag_of_object_tokens=bag_of_object_tokens,
                        max_len=max_len,
                        mask_prob=args.mask_prob,
                        mode=args.pretrain_mode,
                        seed=seed,
                    )
                    generated_data_.append([*outputs, labels])
                outputs, labels = _corrupted_data_processing(generated_data_)
                loop_data.update({'max_len={:d},eps={:.2f}'.format(max_len, eps): (*outputs, labels)})

            else:
                raise NotImplementedError

    # save
    os.makedirs(save_dir, exist_ok=True)
    save_data(save_dir, loop_data, args.pretrain_mode)

    print('Done!')

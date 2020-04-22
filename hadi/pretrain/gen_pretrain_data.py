import os
import sys
sys.path.append("..")

import numpy as np
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
        raise(NotImplementedError)

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
            raise(NotImplementedError)

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
            raise(NotImplementedError)

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

        for tuple_ in list(conversion_dict.keys()):
            detected = False
            if len(tuple_) == 1:
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
            else:
                raise(ValueError('Max entity len should be 3'))

            if detected:
                detected_range_ = range(list(range_)[start_idx], list(range_)[start_idx] + len(tuple_))
                detected_ranges.append((detected_range_, conversion_dict[tuple_]))

    return sorted(detected_ranges, key=lambda tup: tup[0].start)


def get_masked_input(x, ranges, m, unk_id=3):
    extras = []
    for i in m:
        range_ = ranges[i][0]
        x = np.delete(x, range_)
        x = np.insert(x, range_.start, [unk_id] * len(range_))
        extras.append(range(range_.start + 1, range_.stop))
    extras = list(chain(*extras))
    return np.delete(x, extras)



def fix_inputs(x, starting_position_id=1, S=512):
    # pad token_ids so it gets to max
    x = np.pad(x, (0, S - len(x)), constant_values=0)

    # get the obs and act ranges for x
    obs_ranges, act_ranges = _get_ranges([x], config)
    obs_ranges, act_ranges = obs_ranges[0], act_ranges[0]

    # get the new type_ids
    type_ids = []
    for i in range(len(act_ranges)):
        type_ids.extend([config.obs_id] * len(obs_ranges[i]))
        type_ids.extend([config.act_id] * len(act_ranges[i]))
    type_ids.extend([config.obs_id] * (len(obs_ranges[-1])))
    type_ids = np.pad(type_ids, (0, S - len(type_ids)), constant_values=0)

    # get position_ids
    position_ids = np.arange(starting_position_id, starting_position_id + len(x[x > 0]))
    position_ids = np.pad(position_ids, (0, S - len(position_ids)), constant_values=0)

    return (x, type_ids, position_ids)


def generate_corrupted_data(inputs, config, conversion_dict,
                            S=512, mask_prob=0.25, mode='act', seed=665):
    rng = np.random.RandomState(seed)
    token_ids, type_ides, position_ids = inputs

    masked_token_ids = []
    masked_type_ids = []
    masked_position_ids = []
    object_ranges_labels = []

    obs_ranges, act_ranges = _get_ranges(token_ids, config)

    for ii in tqdm(range(len(token_ids))):
        # get ranges for objects of interest
        if mode == 'act':
            detected_ranges = detect_objects(token_ids[ii], act_ranges[ii], conversion_dict)
        elif mode == 'obs':
            detected_ranges = detect_objects(token_ids[ii], obs_ranges[ii], conversion_dict)
        else:
            raise NotImplementedError

        # randomly select form detected objects
        num_ = len(detected_ranges)
        random_numbers = rng.uniform(0, 1, num_)
        m = np.where(random_numbers < mask_prob)[0]
        # mask the selected objects
        x_masked = get_masked_input(token_ids[ii], detected_ranges, m, config.unk_id)
        # fix the masked inputs to have correct length, position and type ids etc
        outputs_ = fix_inputs(x_masked, position_ids[ii][0], S=S)

        masked_token_ids.append(outputs_[0])
        masked_type_ids.append(outputs_[1])
        masked_position_ids.append(outputs_[2])
        object_ranges_labels.append(detected_ranges)

    outputs = (np.array(masked_token_ids), np.array(masked_type_ids), np.array(masked_position_ids))
    return outputs, object_ranges_labels
# ------------------------------------------------------------------------------------------------------- #





# ----------------------------------------------- Save & Load ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #
def save_data(token_ids, type_ids, position_ids,
              labels=None, permutations_used=None, object_ranges_labels=None,
              pretrain_type='ACT_ORDER', S=512, eps=1.00,
              save_dir='/home/hadivafa/Documents/FTWP/processed_data'):
    save_ = os.path.join(save_dir, "DATA.h5")
    f = h5py.File(save_, "w")
    f.create_group(pretrain_type).create_group('S={:d}'.format(S)).create_group('eps={:.2f}'.format(eps))
    subgroup = f[pretrain_type]['S={:d}'.format(S)]['eps={:.2f}'.format(eps)]

    subgroup.create_dataset('token_ids', data=token_ids)
    subgroup.create_dataset('type_ids', data=type_ids)
    subgroup.create_dataset('position_ids', data=position_ids)

    if labels is not None:
        subgroup.create_dataset('labels', data=labels)

    if permutations_used is not None:
        dset = subgroup.create_dataset('permutations_used', (len(permutations_used),),
                                       dtype=h5py.vlen_dtype(np.dtype('int32')))
        for i in range(len(permutations_used)):
            dset[i] = permutations_used[i]

    if object_ranges_labels is not None:
        grp = subgroup.create_group('object_ranges_labels')

        for i in range(len(object_ranges_labels)):
            subgrp = grp.create_group('i={:d}'.format(i))
            for j in range(len(object_ranges_labels[i])):
                subsubgrp = subgrp.create_group('j={:d}'.format(j))
                subsubgrp.create_dataset('object_range', data=object_ranges_labels[i][j][0])
                subsubgrp.create_dataset('object_index', data=object_ranges_labels[i][j][1])

    print('Data saved at {:s}'.format(save_))
    f.close()


def load_data(pretrain_type='ACT_ORDER', S=512, eps=1.00,
              load_dir='/home/hadivafa/Documents/FTWP/processed_data'):
    load_ = os.path.join(load_dir, "DATA.h5")
    print('Loading data from {:s}'.format(load_))
    f = h5py.File(load_, "r")
    subgroup = f[pretrain_type]['S={:d}'.format(S)]['eps={:.2f}'.format(eps)]

    token_ids = np.array(subgroup['token_ids'])
    type_ids = np.array(subgroup['type_ids'])
    position_ids = np.array(subgroup['position_ids'])

    outputs = (token_ids, type_ids, position_ids)

    if 'labels' in subgroup.keys():
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

    if 'object_ranges_labels' in subgroup.keys():
        grp = subgroup['object_ranges_labels']
        object_ranges_labels = []
        for i in range(len(grp)):
            local_list = []
            subgrp = grp['i={:d}'.format(i)]
            for j in range(len(subgrp)):
                subsubgrp = subgrp['j={:d}'.format(j)]
                object_range = list(subsubgrp['object_range'])
                object_range = range(object_range[0], object_range[-1] + 1)
                object_index = np.array(subsubgrp['object_index']).item()
                local_list.append((object_range, object_index))
            object_ranges_labels.append(local_list)
        outputs += (object_ranges_labels,)
        print('object_ranges_labels added to outputs')

    f.close()
    return outputs
# ------------------------------------------------------------------------------------------------------- #





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("pretrain_type", help="(str) pretrain data type", type=str)
    parser.add_argument("--k", help="(integer >= 2) will have k! permutations.  defautl: 3", type=int, default=3)
    parser.add_argument("--mask_prob", help="(float) portion of data to distort.  defautl: 0.25", type=float, default=0.25)
    parser.add_argument("--S", help="(integer) sequence length. default: 512", type=int, default=512)
    parser.add_argument("--eps", help="(float) epsilon. default: 1.00", type=float, default=1.00)
    parser.add_argument("--seed", help="(integer) random seed. default: 665", type=int, default=665)
    parser.add_argument("--dir", help="(str) directory to load and save. default: '/home/hadivafa/Documents/FTWP/processed_data'",
                        type=str, default='/home/hadivafa/Documents/FTWP/processed_data')

    args = parser.parse_args()

    _allowed_pretrain_types = [
        'ACT_ORDER', 'ACT_ENTITY', 'ACT_VERB',
        'OBS_ORDER', 'OBS_ENTITY', 'OBS_VERB',]

    if args.pretrain_type not in _allowed_pretrain_types:
        raise ValueError('enter correct pretrain type.  allowed opetions: \n{}'.format(_allowed_pretrain_types))



    load_ = os.path.join(args.dir, 'data_max_len={:d}.npy'.format(args.S))
    data_all = np.load(load_, allow_pickle=True).item()
    data = data_all['eps={:.2f}'.format(args.eps)]

    token_ids = data['sequence_ids']
    type_ids = data['type_ids']
    segment_ids = data['segment_ids']
    position_ids = data['position_ids']
    masks = data['masks']

    inputs = (token_ids, type_ids, position_ids)
    config = TransformerConfig()

    if args.pretrain_type in ['ACT_ORDER', 'OBS_ORDER']:
        outputs, labels, permutations_used = generate_permutated_data(
            inputs, config, k=args.k, mode=args.pretrain_type[:3].lower())
        save_data(*outputs, labels=labels, permutations_used=permutations_used,
                  pretrain_type=args.pretrain_type, S=args.S, eps=args.eps, save_dir=args.dir)

    elif args.pretrain_type in ['ACT_ENTITY', 'OBS_ENTITY']:
        outputs, object_ranges_labels = generate_corrupted_data(
            inputs, config, data['entity2indx'], S=args.S, mask_prob=args.mask_prob,
            mode=args.pretrain_type[:3].lower(), seed=args.seed)
        save_data(*outputs, object_ranges_labels=object_ranges_labels,
                  pretrain_type=args.pretrain_type, S=args.S, eps=args.eps, save_dir=args.dir)

    elif args.pretrain_type in ['ACT_VERB', 'OBS_VERB']:
        outputs, object_ranges_labels = generate_corrupted_data(
            inputs, config, data['verb2indx'], S=args.S, mask_prob=args.mask_prob,
            mode=args.pretrain_type[:3].lower(), seed=args.seed)
        save_data(*outputs, object_ranges_labels=object_ranges_labels,
                  pretrain_type=args.pretrain_type, S=args.S, eps=args.eps, save_dir=args.dir)
    else:
        raise(NotImplementedError)

    print('Done!')

import os
from os.path import join as pjoin
import re
import spacy
import numpy as np
from collections import Counter


def _exract_data_for_modeling(seq_, segment_, max_len):
    sequences, segments = np.empty((0, max_len)), np.empty((0, max_len)),
    positions, masks = np.empty((0, max_len)), np.empty((0, max_len))
    extracted_data = []

    num = int(np.ceil(len(seq_) / max_len))
    tmp = np.concatenate(
        [np.expand_dims(seq_, 0), np.expand_dims(segment_, 0), np.expand_dims(np.arange(len(seq_)), 0)])

    if num == 1:
        extracted_data.append(tmp)

    elif num == 2:
        # extract first chunk
        first_chunk = tmp[:, :max_len]
        if first_chunk[1, -1] % 2 == 1:
            threshold_id = sorted(np.unique(first_chunk[1, :]))[-1]
        else:
            threshold_id = sorted(np.unique(first_chunk[1, :]))[-2]
        acceptable_indxs = np.where(first_chunk[1] < threshold_id)[0]
        selected = first_chunk[:, acceptable_indxs]
        extracted_data.append(selected)

        # extract last chunk
        last_chunk = tmp[:, -max_len:]
        if last_chunk[1, 0] % 2 == 1:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[0]
        else:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[1]
        acceptable_indxs = np.where(last_chunk[1] > threshold_id)[0]
        selected = last_chunk[:, acceptable_indxs]
        extracted_data.append(selected)

    elif num >= 3:
        for i in range(num - 1):
            first_chunk = tmp[:, :max_len]
            if first_chunk[1, -1] % 2 == 1:
                threshold_id = sorted(np.unique(first_chunk[1, :]))[-1]
            else:
                threshold_id = sorted(np.unique(first_chunk[1, :]))[-2]
            acceptable_indxs = np.where(first_chunk[1] < threshold_id)[0]
            selected = first_chunk[:, acceptable_indxs]
            extracted_data.append(selected)
            if i < num - 2:
                delete_indxs = np.where(first_chunk[1] < threshold_id - 1)[0]
                tmp = np.delete(tmp, delete_indxs, axis=1)

        # extract last chunk
        last_chunk = tmp[:, -max_len:]
        if last_chunk[1, 0] % 2 == 1:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[0]
        else:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[1]
        acceptable_indxs = np.where(last_chunk[1] > threshold_id)[0]
        selected = last_chunk[:, acceptable_indxs]
        extracted_data.append(selected)

    for x in extracted_data:
        seq_arr = np.pad(x[0], (0, max_len - x.shape[1]), constant_values=0)
        seg_arr = np.pad(x[1], (0, max_len - x.shape[1]), constant_values=-1)
        pos_arr = np.pad(x[2], (0, max_len - x.shape[1]), constant_values=-1)
        mask_arr = np.ones(max_len)
        mask_arr[x.shape[1]:] = 0

        sequences = np.concatenate([sequences, np.expand_dims(seq_arr, 0)])
        segments = np.concatenate([segments, np.expand_dims(seg_arr, 0)])
        positions = np.concatenate([positions, np.expand_dims(pos_arr, 0)])
        masks = np.concatenate([masks, np.expand_dims(mask_arr, 0)])

    return sequences, segments, positions, masks


def process_data(data_files, lang_data=None, max_len=512, do_plot=False, verbose=False):
    if type(data_files) is not list:
        data_files = [data_files]

    trajectories = []
    traj_admissible_cmd_pairs = []
    teacher_tuples = []

    unigram_counts = Counter()
    verb_counts = Counter()
    entity_counts = Counter()
    act_counts = Counter()
    walkthroughs_len_counts = Counter()

    w2i = {'<PAD>': 0, '[OBS]': 1, '[ACT]': 2, '<UNK>': 3, '[TRAJ]': 4}
    entity2indx = {}
    verb2indx = {}
    act2indx = {}

    for f in data_files:
        try:
            data_ = np.load(f, allow_pickle=True).item()

            if verbose:
                print('loading .../{}'.format(f.split('/')[-1]))
                print('num trajectories found: ', len(data_['trajectories']))

            trajectories.extend(data_['trajectories'])
            traj_admissible_cmd_pairs.extend(data_['traj_admissible_cmd_pairs'])
            teacher_tuples.extend(data_['teacher_tuples'])

            if lang_data is not None:
                unigram_counts = lang_data['unigram_counts']
                verb_counts = lang_data['verb_counts']
                entity_counts = lang_data['entity_counts']
                act_counts = lang_data['act_counts']

            for k, v in data_['verb_counts'].most_common():
                verb_counts[k] += v
            for k, v in data_['entity_counts'].most_common():
                entity_counts[k] += v
            for k, v in data_['admissible_commands_counts'].most_common():
                act_counts[k] += v
            for k, v in data_['walkthrough_len_counts'].most_common():
                walkthroughs_len_counts[k] += v
        except FileNotFoundError:
            print('File .../{:s} not found'.format(f.split('/')[-1]))

    if lang_data is not None:
        w2i = lang_data['w2i']
        entity2indx = lang_data['entity2indx']
        verb2indx = lang_data['verb2indx']
        act2indx = lang_data['act2indx']

    # get unigrams, w2i and i2w
    for tau in trajectories:
        for tok in tau:
            unigram_counts[tok] += 1

    for tok in unigram_counts:
        if tok not in w2i:
            w2i.update({tok: len(w2i)})

    # update w2i to add extra entities and verbose
    tokenizer = get_nlp().tokenizer

    for entity in list(entity_counts.keys()):
        entity_tokens = preproc(entity, tokenizer)
        for ent_toks in entity_tokens:
            if ent_toks not in w2i:
                w2i.update({ent_toks: len(w2i)})

    for act in list(act_counts.keys()):
        act_tokens = preproc(act, tokenizer)
        for act_toks in act_tokens:
            if act_toks not in w2i:
                w2i.update({act_toks: len(w2i)})

    for verb in list(verb_counts.keys()):
        if verb not in w2i:
            w2i.update({verb: len(w2i)})

    i2w = {w2i[tok]: tok for tok in w2i}

    # Get entity and verb to indx and vice versa
    entities_tokenized = [tuple(w2i[x] for x in preproc(ent, tokenizer)) for ent in list(entity_counts.keys())]
    for ent in entities_tokenized:
        if ent not in entity2indx:
            entity2indx.update({ent: len(entity2indx)})
    indx2entity = {entity2indx[ent]: ent for ent in entity2indx}

    verbs_tokenized = [tuple(w2i[x] for x in preproc(verb, tokenizer)) for verb in list(verb_counts.keys())]
    for verb in verbs_tokenized:
        if verb not in verb2indx:
            verb2indx.update({verb: len(verb2indx)})
    indx2verb = {verb2indx[verb]: verb for verb in verb2indx}

    acts_tokenized = [tuple(['[ACT]'] + preproc(act, tokenizer)) for act in list(act_counts.keys())]
    for act in acts_tokenized:
        if act not in act2indx:
            act2indx.update({act: len(act2indx)})

    # update act2indx if necessary
    for _, cmds in traj_admissible_cmd_pairs:
        cmds_toks = [tuple(['[ACT]'] + preproc(c, tokenizer)) for c in cmds]
        for act in cmds_toks:
            if act not in act2indx:
                act2indx.update({act: len(act2indx)})

    indx2act = {act2indx[act]: act for act in act2indx}

    # Turn string based data into integer based
    trajectory_token_ids = []
    trajectory_segment_ids = []
    for tau in trajectories:
        tau_indxed = []
        seg_indxed = []
        seg_idx = -1
        for token in tau:
            tau_indxed.append(w2i[token])
            if token in ['[OBS]', '[ACT]']:
                seg_idx += 1
            seg_indxed.append(seg_idx)

        trajectory_token_ids.append(tau_indxed)
        trajectory_segment_ids.append(seg_indxed)

    # process the traj/admissible cmds pairs
    processed_traj_adm_cmds = []
    for traj, cmds in traj_admissible_cmd_pairs:
        traj_ids = [w2i[x] for x in traj]
        cmds_toks = [tuple(['[ACT]'] + preproc(c, tokenizer)) for c in cmds]
        cmds_turned_to_indx = [act2indx[item] for item in cmds_toks]

        processed_traj_adm_cmds.append((traj_ids, cmds_turned_to_indx))

    # extract and pad data, ready for modeling
    sequences_all = np.empty((0, max_len - 1))
    segments_all = np.empty((0, max_len - 1))
    positions_all = np.empty((0, max_len - 1))
    masks_all = np.empty((0, max_len - 1))
    for i in tqdm(range(len(trajectory_token_ids))):
        assert len(trajectory_token_ids[i]) == len(trajectory_segment_ids[i]), 'otherwise there is a serious problem'
        seqs_, segs_, poss_, msks_ = _exract_data_for_modeling(trajectory_token_ids[i], trajectory_segment_ids[i],
                                                               max_len - 1)

        sequences_all = np.concatenate([sequences_all, seqs_])
        segments_all = np.concatenate([segments_all, segs_])
        positions_all = np.concatenate([positions_all, poss_])
        masks_all = np.concatenate([masks_all, msks_])

    # plot some histogram
    if do_plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')

        plt.figure(figsize=(16, 3))

        data_to_plot = [len(x) for x in trajectories]
        plt.hist(data_to_plot, bins=100)
        plt.title('{} . . . string length of each trajectory. mean = {:.2f}'
                  .format(data_files[0].split('/')[-2], np.mean(data_to_plot)))
        plt.show()

    # so that the padding index will be 0
    # segment in [1, 1, 1, 2, 2, 3, 3, 3, ..., 0] and positions in
    segments_all += 1
    # position in [1, 2, 3, ..., 0] and positions in
    positions_all += 1

    # make token type ids also
    # type is [1, 1, 1, 2, 2, 1, 1, 1, ..., 0]
    token_types_all = segments_all.astype(int).copy()
    token_types_all[token_types_all > 0] = 1 - ((token_types_all[token_types_all > 0] % 2) - 1)

    # quality control (check if each seq has at least one act and one obs token in it)
    bad_indices = []
    for i in range(len(sequences_all)):
        if not (w2i['[OBS]'] in sequences_all[i] and w2i['[ACT]'] in sequences_all[i]):
            bad_indices.append(i)

    if len(bad_indices) > 0:
        print('Found bad indices: ', bad_indices)

    # delete the bad indices:
    sequences_all = np.delete(sequences_all, bad_indices, axis=0)
    token_types_all = np.delete(token_types_all, bad_indices, axis=0)
    positions_all = np.delete(positions_all, bad_indices, axis=0)
    masks_all = np.delete(masks_all, bad_indices, axis=0)
    segments_all = np.delete(segments_all, bad_indices, axis=0)

    # finally, add the [TRAJ] token to the beginning of all
    _cat_column = np.ones((len(sequences_all), 1))

    sequences_all = np.concatenate((_cat_column * w2i['[TRAJ]'], sequences_all), axis=1)

    token_types_all = np.concatenate((_cat_column, token_types_all), axis=1)
    masks_all = np.concatenate((_cat_column, masks_all), axis=1)
    segments_all = np.concatenate((_cat_column, segments_all), axis=1)

    positions_all[positions_all > 0] += 1
    positions_all = np.concatenate((_cat_column, positions_all), axis=1)

    traj_data = {
        'trajectories': trajectories, 'trajectory_token_ids': trajectory_token_ids,
        'trajectory_segment_ids': trajectory_segment_ids, 'teacher_tuples': teacher_tuples,
        'traj_admissible_cmds_pairs': processed_traj_adm_cmds, 'walkthroughs_len_counts': walkthroughs_len_counts,
        'sequence_ids': sequences_all.astype(int), 'type_ids': token_types_all.astype(int), 'position_ids': positions_all.astype(int),
        'masks': masks_all.astype(int), 'segment_ids': segments_all.astype(int),
    }
    lang_data = {
        'verb_counts': verb_counts, 'entity_counts': entity_counts,
        'unigram_counts': unigram_counts, 'act_counts': act_counts,
        'entity2indx': entity2indx, 'indx2entity': indx2entity,
        'verb2indx': verb2indx, 'indx2verb': indx2verb,
        'act2indx': act2indx, 'indx2act': indx2act,
        'w2i': w2i, 'i2w': i2w,
    }

    return traj_data, lang_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "game_type", help="(str) game type. (e.g. tw_cooking)",
        type=str
    )
    parser.add_argument(
        "max_len", help="max length of trajectories. default: 512",
        type=int, default=512,
    )
    parser.add_argument(
        "--game_spec", help="game specifics such as brief or detailed goal, quest length and so on. default is None",
        type=str, default="",
    )

    args = parser.parse_args()

    import sys
    from tqdm import tqdm
    sys.path.append("..")
    from model.preprocessing import get_nlp, preproc
    from model.configuration import DataConfig

    data_config = DataConfig(game_type=args.game_type, game_spec=args.game_spec, train_valid_test=True)
    _game_variants = ['train', 'valid', 'test']

    lang_save_dir = data_config.lang_dir
    os.makedirs(lang_save_dir, exist_ok=True)

    for gvar_id, game_var in enumerate(_game_variants):
        load_dir = pjoin(data_config.base_dirs[gvar_id], 'raw_trajectories')
        traj_save_dir = data_config.processed_dirs[gvar_id]
        assert (game_var in load_dir) and (game_var in traj_save_dir), "..."

        lang_dir_file = pjoin(lang_save_dir, 'lang_data_max_len={:d}.npy'.format(args.max_len))
        try:
            lang_data_all = np.load(lang_dir_file, allow_pickle=True).item()
        except FileNotFoundError:
            lang_data_all = {}

        traj_data_all = {}

        for eps in tqdm(np.arange(0.0, 2, 1), desc='processing {:s}, S={:d}'.format(game_var, args.max_len)):
            dir_ = pjoin(load_dir, 'eps={:.2f}'.format(eps))
            files_ = os.listdir(dir_)
            data_files = [pjoin(dir_, x) for x in sorted(files_)]

            try:
                current_lang_data = lang_data_all['eps={:.2f}'.format(eps)]
            except KeyError:
                current_lang_data = None

            traj_data_, lang_data_ = process_data(
                data_files=data_files,
                lang_data=current_lang_data,
                max_len=args.max_len,
                do_plot=False, verbose=False)
            traj_data_all.update({'eps={:.2f}'.format(eps): traj_data_})
            lang_data_all.update({'eps={:.2f}'.format(eps): lang_data_})

        os.makedirs(traj_save_dir, exist_ok=True)
        traj_dir_file = pjoin(traj_save_dir, 'traj_data_max_len={:d}.npy'.format(args.max_len))

        np.save(traj_dir_file, traj_data_all)
        np.save(lang_dir_file, lang_data_all)

    print('Done!')

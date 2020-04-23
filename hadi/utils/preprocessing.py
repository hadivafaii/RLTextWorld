import numpy as np
import os
import re
import spacy
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')



def get_nlp():
    """
    get spacy nlp and modify its tokenizer
    """
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    # remove dash (-) from infixes
    infixes = list(nlp.Defaults.infixes).copy()
    del infixes[6]
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer
    return nlp



def preproc(string, tokenizer):
    """
    basically to tokenize
    """
    if string is None:
        return [None]

    pattern = r"[_\\|/$>,.!?>]"
    s = re.sub(pattern, '', string).replace("\n", ' ').strip().lower()

    if '-=' in s:
        pattern = r"""-= [\s\S]* =-"""
        m = re.search(pattern, s)
        span = m.span()
        s = s[:span[0]] + m.group(0).replace(' ', '') + s[span[1]:]
    if '***' in s:
        pattern = r"""\*\*\*[\s\S]*\*\*\*"""
        m = re.search(pattern, s)
        span = m.span()
        s = s[:span[0]] + m.group(0).replace('*', '-').replace(' ', '-') + s[span[1]:]

    return [t.text for t in tokenizer(s) if not t.is_space]



def _exract_data_for_modeling(seq_, segment_, S):
    sequences, segments = np.empty((0, S)), np.empty((0, S)),
    positions, masks = np.empty((0, S)), np.empty((0, S))
    extracted_data = []

    num = int(np.ceil(len(seq_) / S))
    tmp = np.concatenate([np.expand_dims(seq_, 0), np.expand_dims(segment_, 0), np.expand_dims(np.arange(len(seq_)), 0)])

    if num == 1:
        extracted_data.append(tmp)

    elif num == 2:
        # extract first chunk
        first_chunk = tmp[:, :S]
        if first_chunk[1, -1] % 2 == 1:
            threshold_id = sorted(np.unique(first_chunk[1, :]))[-1]
        else:
            threshold_id = sorted(np.unique(first_chunk[1, :]))[-2]
        acceptable_indxs = np.where(first_chunk[1] < threshold_id)[0]
        selected = first_chunk[:, acceptable_indxs]
        extracted_data.append(selected)

        # extract last chunk
        last_chunk = tmp[:, -S:]
        if last_chunk[1, 0] % 2 == 1:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[0]
        else:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[1]
        acceptable_indxs = np.where(last_chunk[1] > threshold_id)[0]
        selected = last_chunk[:, acceptable_indxs]
        extracted_data.append(selected)

    elif num >= 3:
        for i in range(num - 1):
            first_chunk = tmp[:, :S]
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
        last_chunk = tmp[:, -S:]
        if last_chunk[1, 0] % 2 == 1:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[0]
        else:
            threshold_id = sorted(np.unique(last_chunk[1, :]))[1]
        acceptable_indxs = np.where(last_chunk[1] > threshold_id)[0]
        selected = last_chunk[:, acceptable_indxs]
        extracted_data.append(selected)


    for x in extracted_data:
        seq_arr = np.pad(x[0], (0, S - x.shape[1]), constant_values=0)
        seg_arr = np.pad(x[1], (0, S - x.shape[1]), constant_values=-1)
        pos_arr = np.pad(x[2], (0, S - x.shape[1]), constant_values=-1)
        mask_arr = np.ones(S)
        mask_arr[x.shape[1]:] = 0

        sequences = np.concatenate([sequences, np.expand_dims(seq_arr, 0)])
        segments = np.concatenate([segments, np.expand_dims(seg_arr, 0)])
        positions = np.concatenate([positions, np.expand_dims(pos_arr, 0)])
        masks = np.concatenate([masks, np.expand_dims(mask_arr, 0)])

    return sequences, segments, positions, masks


def process_data(data_files, max_length=512, do_plot=True, verbose=False):
    if type(data_files) is not list:
        data_files = [data_files]

    trajectories = []
    verb_counts = Counter()
    entity_counts = Counter()
    walkthroughs_len_counts = Counter()

    for f in data_files:
        try:
            data_ = np.load(f, allow_pickle=True).item()

            if verbose:
                print('loading .../{}'.format(f.split('/')[-1]))
                print('num trajectories found: ', len(data_['trajectories']))

            trajectories.extend(data_['trajectories'])
            for k, v in data_['verb_counts'].most_common():
                verb_counts[k] += v
            for k, v in data_['entity_counts'].most_common():
                entity_counts[k] += v
            for k, v in data_['walkthrough_len_counts'].most_common():
                walkthroughs_len_counts[k] += v
        except FileNotFoundError:
            print('File .../{:s} not found'.format(f.split('/')[-1]))


    ### get unigrams, w2i and i2w
    unigram_counts = Counter()
    for tau in trajectories:
        for tok in tau:
            unigram_counts[tok] += 1

    w2i = {'<PAD>': 0, '[OBS]': 1, '[ACT]': 2, '<UNK>': 3}
    for tok in unigram_counts:
        if tok not in w2i:
            w2i.update({tok: len(w2i)})

    # first update w2i to add extra verbose
    for verb in list(verb_counts.keys()):
        if verb not in w2i:
            w2i.update({verb: len(w2i)})

    i2w = {w2i[tok]: tok for tok in w2i}


    ### Get entity and verb to indx and vice versa
    entities_tokenized = [tuple(w2i[x] for x in preproc(ent, get_nlp().tokenizer)) for ent in list(entity_counts.keys())]
    entity2indx = {}
    for ent in entities_tokenized:
        entity2indx.update({ent: len(entity2indx)})
    indx2entity = {entity2indx[ent]: ent for ent in entity2indx}

    verbs_tokenized = [tuple(w2i[x] for x in preproc(verb, get_nlp().tokenizer)) for verb in list(verb_counts.keys())]
    verb2indx = {}
    for verb in verbs_tokenized:
        verb2indx.update({verb: len(verb2indx)})
    indx2verb = {verb2indx[verb]: verb for verb in verb2indx}




    ### Turn string based data into integer based
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


    ### extract and pad data, ready for modeling
    sequences_all = np.empty((0, max_length))
    segments_all = np.empty((0, max_length))
    positions_all = np.empty((0, max_length))
    masks_all = np.empty((0, max_length))
    for i in range(len(trajectory_token_ids)):
        assert len(trajectory_token_ids[i]) == len(trajectory_segment_ids[i]), 'otherwise there is a serious problem'
        seqs_, segs_, poss_, msks_ = _exract_data_for_modeling(trajectory_token_ids[i], trajectory_segment_ids[i], max_length)

        sequences_all = np.concatenate([sequences_all, seqs_])
        segments_all = np.concatenate([segments_all, segs_])
        positions_all = np.concatenate([positions_all, poss_])
        masks_all = np.concatenate([masks_all, msks_])

        print('[PROGRESS]   . . .   %0.2f %s done' % (100 * (i + 1) / len(trajectory_token_ids), '%'), end='\r')


    ### plot some histogram
    if do_plot:
        plt.figure(figsize=(16, 3))

        data_to_plot = [len(x) for x in trajectories]
        plt.hist(data_to_plot, bins=100)
        plt.title('{} . . . string length of each trajectory. mean = {:.2f}'
                    .format(data_files[0].split('/')[-2], np.mean(data_to_plot)))
        plt.show()

    ### so that the padding index will be 0
    # segment in [1, 1, 1, 2, 2, 3, ..., 0] and positions in
    segments_all += 1
    # position in [1, 2, 3, ..., 0] and positions in
    positions_all += 1

    # make token type ids also
    token_types_all = segments_all.astype(int).copy()
    token_types_all[token_types_all > 0] = 1 - ((token_types_all[token_types_all > 0] % 2) - 1)

    data = {
        'trajectories': trajectories, 'trajectory_token_ids': trajectory_token_ids, 'trajectory_segment_ids': trajectory_segment_ids,
        'sequence_ids': sequences_all.astype(int), 'type_ids': token_types_all.astype(int),
        'position_ids': positions_all.astype(int), 'masks': masks_all.astype(int),
        'segment_ids': segments_all.astype(int), 'walkthroughs_len_counts': walkthroughs_len_counts,
        'verb_counts': verb_counts, 'entity_counts': entity_counts, 'unigram_counts': unigram_counts,
        'entity2indx': entity2indx, 'indx2entity': indx2entity,
        'verb2indx': verb2indx, 'indx2verb':indx2verb,
        'w2i': w2i, 'i2w': i2w,
    }

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "game_type", help="(str) game type. (e.g. tw_cooking/train)",
        type=str
    )
    parser.add_argument(
        "max_length", help="max length of trajectories. default: 512",
        type=int, default=512,
    )
    parser.add_argument(
        "--load_dir", help="save directory. default: '~/game_type/raw_trajectories'",
        type=str, default="raw_trajectories",
    )
    parser.add_argument(
        "--save_dir", help="save directory. default: '~/game_type/processed_trajectories'",
        type=str, default="processed_trajectories",
    )

    args = parser.parse_args()

    base_dir = os.path.join('/home/hadivafa/Documents/FTWP/trajectories', args.game_type)

    args.load_dir = os.path.join(base_dir, args.load_dir)
    args.save_dir = os.path.join(base_dir, args.save_dir)

    data_all = {}

    import os
    from tqdm import tqdm
    for eps in tqdm(np.arange(0.0, 1.1, 0.1), desc='processing... S={:d}'.format(args.max_length)):
        dir_ = os.path.join(args.load_dir, 'eps={:.2f}'.format(eps))
        files_ = ['iter={}.npy'.format(x) for x in np.arange(20)]
        data_files = [os.path.join(dir_, x) for x in files_]

        data_ = process_data(data_files, max_length=args.max_length)
        data_all.update({'eps={:.2f}'.format(eps): data_})

    os.makedirs(args.save_dir, exist_ok=True)
    dir_ = os.path.join(args.save_dir, 'max_len={:d}.npy'.format(args.max_length))
    np.save(dir_, data_all)

    print('Done!')

import os
import numpy as np

from time import time
from tqdm import tqdm
from utils import convert_time

from textworld import EnvInfos
from textworld.gym import envs
from collections import Counter


def generate_trajectory(game_files, tokenizer, max_steps=100, episodes=50,
                        batch_size=111, mode='walkthrough', epsilon=1, seed=0):
    rng = np.random.RandomState(seed)

    requested_infos = EnvInfos(
        max_score=True, verbs=True, entities=True, moves=True,
        admissible_commands=True, policy_commands=True, intermediate_reward=True, game=True)

    env = envs.textworld_batch.TextworldBatchGymEnv(
        game_files, request_infos=requested_infos,
        batch_size=batch_size, max_episode_steps=max_steps,
        auto_reset=False, asynchronous=False)

    # initialize the datas you want to save
    all_trajectories = list()
    all_teacher_tuples = list()
    verb_counts = Counter()
    entity_counts = Counter()
    walkthroughs_len_counts = Counter()

    # Get trajectories
    for _ in tqdm(range(episodes), desc='extracting trajectories'):
        obs, infos = env.reset()

        verbs = [vrb for game_verbs in infos['verbs'] for vrb in game_verbs]
        entities = [ent for game_entities in infos['entities'] for ent in game_entities]
        walkthroughs = [game.walkthrough for game in infos['game']]
        walkthroughs_len = [len(x) for x in walkthroughs]

        for vrb in verbs:
            verb_counts[vrb] += 1
        for ent in entities:
            entity_counts[ent] += 1
        for item in walkthroughs_len:
            walkthroughs_len_counts[item] += 1

        admissible_commands = [
            [cmd for cmd in cmd_list if cmd not in ['look']]
            for cmd_list in infos["admissible_commands"]]

        if mode == 'policy' and min([len(x) for x in infos["policy_commands"]]) == 0:
            print('warning, policy mode activated but no policy commands found')

        trajectory = [['[OBS]'] + preproc(x, tokenizer) for x in obs]
        teacher_tuples = [list() for _ in range(batch_size)]

        nb_moves_this_episode = [0] * batch_size
        dones = [False] * batch_size
        trajectory_dones = [False] * batch_size
        while not all(dones):
            random_numbers = rng.uniform(0, 1, batch_size)
            if mode == 'walkthrough':
                walkthrough_commands = [tup[1][min(len(tup[1]) - 1, tup[0])]
                                        for tup in zip(nb_moves_this_episode, walkthroughs)]
                commands = []
                for i in range(batch_size):
                    if epsilon >= random_numbers[i]:
                        commands.append(walkthrough_commands[i])
                        nb_moves_this_episode[i] += 1
                    else:
                        commands.append(rng.choice(admissible_commands[i]))
                        nb_moves_this_episode[i] = rng.choice(range(2 * (nb_moves_this_episode[i] // 3),
                                                                    nb_moves_this_episode[i] + 1))
            elif mode == 'policy':
                policy_commands = infos['policy_commands']
                commands = []
                for i in range(batch_size):
                    if epsilon >= random_numbers[i]:
                        try:
                            commands.append(policy_commands[i][0])
                        except IndexError:
                            commands.append([])
                    else:
                        commands.append(rng.choice(admissible_commands[i]))
            else:
                raise NotImplementedError

            obs, scores, dones, infos = env.step(commands)

            for i in range(batch_size):
                if not trajectory_dones[i]:
                    trajectory[i] = (trajectory[i] +
                                     ['[ACT]'] + preproc(commands[i], tokenizer) +
                                     ['[OBS]'] + preproc(obs[i], tokenizer))

                    # create teacher tuples
                    if mode == 'policy':
                        if infos['intermediate_reward'][i] == 1:
                            correct_command = commands[i]
                        else:
                            correct_command = infos['policy_commands'][i][0]
                        teacher_tuples[i].append(
                            (
                                preproc(commands[i], tokenizer),
                                infos['intermediate_reward'][i],
                                preproc(correct_command, tokenizer)
                            )
                        )

                    if dones[i]:
                        trajectory_dones[i] = True

        all_trajectories.extend(trajectory)
        all_teacher_tuples.extend(teacher_tuples)

    data = {'trajectories': all_trajectories, 'teacher_tuples': all_teacher_tuples,
            'verb_counts': verb_counts, 'entity_counts': entity_counts,
            'walkthrough_len_counts': walkthroughs_len_counts}

    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("game_type", help="(str) game type. (e.g. tw_cooking/train)", type=str)
    parser.add_argument("iter", help="iteration step", type=int)
    parser.add_argument("num_groups", help="how many subgroups to divide to", type=int)

    parser.add_argument(
        "--exploration_mode", help="exploration mode: policy or walkthrough. default: walkthrough",
        type=str, default='walkthrough',
        )
    parser.add_argument(
        "--epsilon", help="walkthrough to random cutoff portion. default: 1",
        type=float, default=1.0,
        )
    parser.add_argument(
        "--max_steps", help="maximum number of steps per episode. default: 100",
        type=int, default=100,
        )
    parser.add_argument(
        "--extra_episodes", help="number of extra episodes per game. default: 1",
        type=int, default=1,
        )
    parser.add_argument(
        "--batch_size", help="batch_size. defaults to 32",
        type=int, default=32,
        )
    parser.add_argument(
        "--seed", help="Random seed. default: 665",
        type=int, default=665,
        )
    parser.add_argument(
        "--game_spec", help="game_spec code such as dd or b-large etc. default is None",
        type=str, default="",
    )
    parser.add_argument(
        "--silent", help="add this flag to force silence output verbosity",
        action="store_true",
        )

    args = parser.parse_args()

    if not args.silent:
        print("verbosity is on")

    assert 0 <= args.epsilon <= 1, 'epsilon is a float in [0, 1]'

    import sys
    sys.path.append("..")
    from model.preprocessing import get_nlp, preproc
    from model.configuration import DataConfig

    data_config = DataConfig(game_type=args.game_type, game_spec=args.game_spec)

    load_dir = data_config.games_dir
    save_dir = os.path.join(data_config.base_dir, 'raw_trajectories')

    game_files = os.listdir(load_dir)
    game_files = [os.path.join(load_dir, g) for g in game_files if '.ulx' in g]

    group_size = int(np.ceil(len(game_files) / args.num_groups))

    a = args.iter * group_size
    b = (args.iter + 1) * group_size

    game_files = game_files[a:b]
    num_games = len(game_files)

    episodes = int(np.ceil(num_games / args.batch_size))
    episodes *= args.extra_episodes

    if not args.silent:
        msg = '[PROGRESS] this is iter # {:d}'
        print(msg.format(args.iter))
        msg = '[PROGRESS] data generation initiated using games {:d}:{:d}.   mode: {:s},   epsilon: {:f}'
        print(msg.format(a, b, args.exploration_mode, args.epsilon))
        msg = '[PROGRESS] max_steps: {:d},   episodes: {:d},   batch_size: {:d}'
        print(msg.format(args.max_steps, episodes, args.batch_size))

    tokenizer = get_nlp().tokenizer

    start_time = time()
    raw_trajectories = generate_trajectory(
        game_files, tokenizer,
        max_steps=args.max_steps, episodes=episodes, batch_size=args.batch_size,
        mode=args.exploration_mode, epsilon=args.epsilon, seed=args.seed)
    end_time = time()

    # save data
    save_ = os.path.join(save_dir, 'eps={:.2f}'.format(args.epsilon))
    os.makedirs(save_, exist_ok=True)
    file_name = 'iter={:d}.npy'.format(args.iter)
    np.save(
        os.path.join(save_, file_name),
        raw_trajectories)

    if not args.silent:
        print('[PROGRESS] done!')
        print('[PROGRESS] file saved: ', file_name)

    convert_time(end_time - start_time)

    print('Done!')

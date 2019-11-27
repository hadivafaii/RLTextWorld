import os
import numpy as np
from glob import glob
from collections import defaultdict

import gym
import textworld.gym


def play(agent, path, max_step=100, nb_episodes=10, verbose=True):
    infos_to_request = agent.infos_to_request
    infos_to_request.max_score = True  # Needed to normalize the scores.

    gamefiles = [path]
    if os.path.isdir(path):
        gamefiles = glob(os.path.join(path, "*.ulx"))

    env_id = textworld.gym.register_games(gamefiles,
                                          request_infos=infos_to_request,
                                          max_episode_steps=max_step)
    env = gym.make(env_id)  # Create a Gym environment to play the text game.

    if verbose:
        if os.path.isdir(path):
            print(os.path.dirname(path), end="")
        else:
            print(os.path.basename(path), end="")  # Collect some statistics: nb_steps, final reward.

    avg_moves, avg_scores, avg_norm_scores = [], [], []
    play_stats = defaultdict()
    scores_dict = defaultdict(list)
    for no_episode in range(nb_episodes):
        obs, infos = env.reset()  # Start new episode.

        score = 0
        done = False
        nb_moves = 0
        scores_ = []
        while not done:
            command = agent.act(obs, score, done, infos)
            obs, score, done, infos = env.step(command)

            scores_.append(score)
            nb_moves += 1

        scores_dict['episode_%d' % no_episode] = scores_
        agent.act(obs, score, done, infos)  # Let the agent know the game is done.

        if verbose:
            print(".", end="")
        avg_moves.append(nb_moves)
        avg_scores.append(score)
        avg_norm_scores.append(score / infos["max_score"])

    play_stats["scores"] = scores_dict
    play_stats["max_score"] = infos["max_score"]
    play_stats["nb_episodes"] = nb_episodes
    play_stats["max_step"] = max_step

    env.close()
    msg = "  \tavg. steps: {:5.1f}; avg. score: {:4.1f} / {}."
    if verbose:
        if os.path.isdir(path):
            print(msg.format(np.mean(avg_moves), np.mean(avg_norm_scores), 1))
        else:
            print(msg.format(np.mean(avg_moves), np.mean(avg_scores), infos["max_score"]))

    return play_stats
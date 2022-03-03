import matplotlib.pyplot as plt
from IPython import display
import json
import os

plt.ion()


def plot(scores, mean_scores, save=1, file_name='scores.txt'):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim()
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    if save:
        scores_folder_path = 'graphs'
        if not os.path.exists(scores_folder_path):
            os.makedirs(scores_folder_path)

        file_name = os.path.join(scores_folder_path, file_name)

        with open(file_name, "w") as fp:
            json.dump(list(zip(scores, mean_scores)), fp)
import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# To keep updating the plot in the SAME figure object
def plot_all_scores(scores, mean_scores):
    display.clear_output(wait = True)
    display.display(plt.gcf())
    plt.clf() # Clear current figure
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0) # Only limiting the min value
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
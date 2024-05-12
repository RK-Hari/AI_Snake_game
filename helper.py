import matplotlib.pyplot as plt
from IPython import display

# Turns on interactive mode in Matplotlib
plt.ion()

def plot(scores, mean_scores):
    #Clears the output of the previous plot to update the display with the new plot.
    display.clear_output(wait=True)
    # Displays the current figure (plot) in the IPython environment.
    display.display(plt.gcf())
    # Clears the current figure (plot) to prepare for a new plot.
    plt.clf()
    # Setting of the title of the graph
    plt.title('Snake Progress Visualisation')
    #X-Label
    plt.xlabel('Number of Games')
    #Y-Label
    plt.ylabel('Score')
    #Plot these figures in the graph
    plt.plot(scores)
    plt.plot(mean_scores)
    #Sets the minimum value for the y-axis to 0, ensuring that the plot starts from 0.
    plt.ylim(ymin=0)
    #Adds text annotation to the plot at the last data point of scores, displaying the score value at that point.
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    #Adds text annotation to the plot at the last data point of mean_scores, displaying the mean score value at that point.
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    #Displays the updated plot. 
    # The argument block=False ensures that the code execution continues without waiting for the plot window to be closed.
    plt.show(block=False)
    # Pauses the execution for a short duration (0.1 seconds) to allow the plot to be displayed and updated smoothly.
    plt.pause(.1)

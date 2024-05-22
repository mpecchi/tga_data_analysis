import matplotlib.pyplot as plt

# Create an empty plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Create an empty line object

# Update function to update the plot
def update_plot(x_data, y_data):
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.relim()  # Update the limits of the axes
    ax.autoscale_view()  # Autoscale the view
    fig.canvas.draw()  # Redraw the canvas
    fig.canvas.flush_events()  # Flush the GUI events
x_data=[]
y_data=[]
# Your for loop
for i in range(10):
    x_data.append(i)
    y_data.append(i**2)
    update_plot(x_data, y_data)
plt.show()
import matplotlib.pyplot as plt     

def plot_2D_start_end(start_x, start_y, end_x, end_y, target_x, target_y, plot_points = 10):
    plt.figure()

    plt.scatter(start_x[:plot_points], start_y[:plot_points])
    plt.scatter(end_x[:plot_points], end_y[:plot_points])
    dx = end_x[:plot_points]-start_x[:plot_points]
    dy = end_y[:plot_points]-start_y[:plot_points]
    plt.quiver(start_x[:plot_points], start_y[:plot_points], dx, dy, scale=1, scale_units='xy', angles='xy', color='r', width=0.005)

    plt.scatter(target_x[:], target_y[:], s=1)
    plt.gca().set_aspect('equal')
    plt.show()
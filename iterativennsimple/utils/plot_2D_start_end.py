import matplotlib.pyplot as plt     

def plot_2D_start_end(start, end, target, plot_points = 10):
    plt.figure()

    plt.scatter(start[:plot_points,0], start[:plot_points,1])
    plt.scatter(end[:plot_points,0], end[:plot_points,1])
    dx = end[:plot_points,0]-start[:plot_points,0]
    dy = end[:plot_points,1]-start[:plot_points,1]
    plt.quiver(start[:plot_points,0], start[:plot_points,1], dx, dy, scale=1, scale_units='xy', angles='xy', color='r', width=0.005)

    plt.scatter(target[:,0], target[:,1], s=1)
    plt.gca().set_aspect('equal')
    plt.show()
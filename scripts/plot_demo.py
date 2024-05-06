# %%
import wandb
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# %%
# Initialize the run
run = wandb.init(entity='rcpaffenroth-wpi',
                 project='plot-test',
                 name='example1')

# %%
# Some basic configuration
wandb.config['test']=1234
wandb.config['bar']='foo'
wandb.config['test2']=56789

# %%
# a table of data
x_values = np.linspace(0, 2*np.pi, 100)
y_values = np.sin(x_values)
data = [[x, y] for (x, y) in zip(x_values, y_values)]
data_table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log(
    {"my_custom_plot_id" : wandb.plot.line(data_table, "x", "y",
           title="Custom Y vs X Line Plot")})

# %%
# an image
image_array = np.random.randn(100, 100)
images = wandb.Image(
    image_array, 
    caption="Top: Output, Bottom: Input"
    )
          
wandb.log({"examples": images})

# %%
# A basic plotly example
# Create a sample dataset
data = [go.Scatter(x=[1, 2, 3], y=[1, 3, 2])]

# Create a Plotly figure
fig = go.Figure(data=data)

# Log the Plotly figure to WandB
wandb.log({"plot": fig})

# %%
# a fancier plotly plot
data=[[1, 25, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, 5, 20]]
fig = px.imshow(data,
                labels=dict(x="Day of Week", y="Time of Day", color="Productivity"),
                x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                y=['Morning', 'Afternoon', 'Evening']
               )
fig.update_xaxes(side="top")
# Log the Plotly figure to WandB
wandb.log({"plot 2": fig})

# %% 
# 3D plot
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

fig = go.Figure(data=[go.Surface(z=z_data.values)])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

# Log the Plotly figure to WandB
wandb.log({"3D plot": fig})

# %%
# A table of images
# Create a table with two columns: "Image" and "Label"
table = wandb.Table(["Image", "Label"])

# Add images to the table
for i in range(4):
    image_array = np.random.randn(100, 100)
    image = wandb.Image(
        image_array, 
        caption="Top: Output, Bottom: Input"
        )
    label = "Label {}".format(i)
    table.add_data(image, label)

# Log the table to WandB
wandb.log({"Image Table": table})


# %%
# A table of plots
# Create a table with two columns: "Image" and "Label"
table = wandb.Table(["Plot", "Label"])

# Add images to the table
for i in range(4):
    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

    fig = go.Figure(data=[go.Surface(z=z_data.values)])

    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90))
    # Create path for Plotly figure
    path_to_plotly_html = "./plotly_figure.html"

    # Write Plotly figure to HTML
    # Set auto_play to False prevents animated Plotly charts
    # from playing in the table automatically
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Add Plotly figure as HTML file into Table
    fig_html = wandb.Html(path_to_plotly_html)

    label = "Label {}".format(i)
    table.add_data(fig_html, label)

# Log the table to WandB
wandb.log({"Plot Table": table})

# %%
# log series



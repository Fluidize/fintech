import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tqdm import tqdm
from plotly.subplots import make_subplots
import scipy.stats as stats
import matplotlib.pyplot as plt

def wiener_process(n_steps, time_unit=1, gbm=True):
    #set w to 1 and multiply by delta_w to get log normal distribution
    y = []
    w = 1 if gbm else 0

    for i in range(n_steps * time_unit):
        delta_t = time_unit / n_steps
        # in a wiener process, the variance of the increment delta_w is delta_t. in order to get the stddev, we sqrt
        # the variance of each step is: Var[ΔW]=Δt
        delta_w = np.random.normal(1 if gbm else 0, np.sqrt(delta_t)) #numpy requires stddev, not variance
        w = (w * delta_w) if gbm else (w + delta_w)
        y.append(w)
    return y

def monte_carlo_simulation(n_steps, n_paths, time_unit=1):
    # if process is log-normal, taking a logarithm reverses the exponential transformation, turning it into a normal distribution
    paths = []
    
    variance = time_unit / n_steps
    stddev = np.sqrt(variance)

    for i in tqdm(range(n_paths), desc="Simulating paths"):
        paths.append(wiener_process(n_steps, time_unit=time_unit, gbm=True))
    
    paths_array = np.array(paths)
    
    df = pd.DataFrame(paths_array.T) 
    return df

def plot_paths(df):
    mean_path = df.mean(axis=1)
    stddev_path = df.std(axis=1)

    fig = make_subplots(rows=2, cols=1, subplot_titles=("Simulated Paths", "Distributions"))

    for i in range(df.shape[1]):
        fig.add_trace(go.Scatter(x=df.index, y=df[i], mode='lines', name=f'Path {i+1}'), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=mean_path, mode='lines', name='Mean Path', line=dict(color='blue', width=2)), row=1, col=1)

    final_values = df.iloc[-1]
    fig.add_trace(go.Histogram(x=final_values, name='Final Values', opacity=0.75, nbinsx=1000), row=2, col=1)

    fig.update_layout(
        title=f'Monte Carlo Simulation | {df.shape[1]} paths | {stddev_path.iloc[0]:.2f} Standard Deviation',
        xaxis_title='Steps',
        yaxis_title='Process Value',
        template='plotly_dark'
    )

    fig.show()

    plt.figure(figsize=(8, 6))
    stats.probplot(np.log(final_values), dist="norm", plot=plt)
    plt.title('Q-Q Plot of Log-Transformed Final Values')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    paths = monte_carlo_simulation(n_steps=1000, n_paths=1000, time_unit=1)
    plot_paths(paths)
CFG = {
    # simulation parameters
    'task' : 'driverextended', # the experiment type. Available are driver, driverextended and fetch
    'sigma_values': [.1, .3],
    'alpha_values': [.5, 1.0, .75, .25],
    'slider_step_size' : [.1, 1.0, 2.0],
    'acquisitions' : ['random','regret','information'],

    # plotting parameters
    'path' : 'simulation_data/driver',
    'sigma_plot' : 0.3
}

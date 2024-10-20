from src.utils.logger import logger

import nevergrad as ng
from src.train.trainer import TrainerWrapper

class HPO:
    def __init__(self, config):
        self.config = config
        self.logger.info("HPO initialized with config.")


    def optimize_hyperparameters(self):
        self.logger.info("Starting hyperparameter optimization...")

        # Extract HPO parameters from config
        lr_range = self.config['hpo_params']['lr_range']
        batch_size_range = self.config['hpo_params']['batch_size_range']
        optimizers = self.config['hpo_params']['optimizers']
        budget = self.config['hpo_params']['budget']

        # Define the search space for the hyperparameters based on config
        param_space = ng.p.Dict(
            learning_rate=ng.p.Log(lower=lr_range[0], upper=lr_range[1]),
            batch_size=ng.p.Scalar(lower=batch_size_range[0], upper=batch_size_range[1]).set_integer_casting(),
            optimizer_name=ng.p.Choice(optimizers)
        )

        # Define the optimizer (you can change this to another optimizer from Nevergrad)
        optimizer = ng.optimizers.OnePlusOne(parametrization=param_space, budget=budget)

        # Define an objective function for Nevergrad to minimize
        def objective_function(ng_params):
            # Update the config with dynamic parameters from Nevergrad
            self.config['training']['lr'] = ng_params['learning_rate']
            self.config['training']['batch_size'] = ng_params['batch_size']
            self.config['training']['optimizer'] = ng_params['optimizer_name']
            
            # Train and return validation loss
            trainer_wrapper = TrainerWrapper(self.config)
            return trainer_wrapper.train()

        # Run the optimization
        recommendation = optimizer.minimize(objective_function)

        # Best hyperparameters found
        print("Best hyperparameters:", recommendation.value)

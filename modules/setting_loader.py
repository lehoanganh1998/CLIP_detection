import yaml

class Setting:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            parameters = yaml.safe_load(file)
        self.model_name = parameters['model']
        self.random_seed = parameters['random_seed']
        self.k_shots_list = parameters['k_shots']
        self.epoch_num = parameters['epoch']
        self.train_location = parameters['train_loc']
        self.test_location = parameters['test_loc']
        self.weight_location = parameters['weight_loc']
        self.prompts = parameters['prompts']
        self.result_file = parameters['result_name']
        self.result_loc = parameters['result_loc']
class BasicState:
    def __init__(self,
                 dataset=None,
                 output_dir=None,
                 sampling_algorithm_data_tab=None,
                 sampling_algorithm_experiments_tab=None,
                 normal_classify_thread_finished=None,
                 resample_classify_thread_finished=None,
                 classified_data_normal_case=None,
                 classified_data_resampled_case=None,
                 classification_algorithm=None,
                 number_of_folds=10,
                 number_of_runs=0):
        self.dataset = dataset
        self.output_dir = output_dir
        self.sampling_algorithm_data_tab = sampling_algorithm_data_tab
        self.sampling_algorithm_experiments_tab = sampling_algorithm_experiments_tab
        self.normal_classify_thread_finished = normal_classify_thread_finished
        self.resample_classify_thread_finished = resample_classify_thread_finished
        self.classified_data_normal_case = classified_data_normal_case
        self.classified_data_resampled_case = classified_data_resampled_case
        self.classification_algorithm = classification_algorithm
        self.number_of_folds = number_of_folds
        self.number_of_runs = number_of_runs

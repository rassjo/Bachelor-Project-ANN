import ANN as ann
import numpy as np
import activation_functions as act
import synthetic_data_generation as sdg
import matplotlib.pyplot as plt
import classification_statistics as cs
from id_generator import best_hash
import txt_utils as tu
import rng_utils as ru

class variable_hp:
    def __init__(self, name, range, is_random_dist = True, num_parts = None, is_log_dist = False, is_rev_open = False, make_int = False):
        self.name = name
        self.range = range
        self.is_random_dist = is_random_dist
        self.num_parts = num_parts
        self.part_index = 0
        self.is_log_dist = is_log_dist
        self.is_rev_open = is_rev_open
        self.make_int = make_int
        self.val = None

    def pick_from_grid(self):
        if self.range[0] != self.range[1]:
            if (self.is_log_dist):
                # Partition the range
                val_pow_range = np.log10(self.range)
                len_pow_parts = val_pow_range[1] - val_pow_range[0]
                len_pow_part = len_pow_parts / (self.num_parts - 1)
                pow_parts = []
                for i in range(0, self.num_parts):
                    pow_part = len_pow_part * i
                    pow_part += val_pow_range[0]
                    pow_parts.append(pow_part)
                val_pow = pow_parts[self.part_index]
                val = 10**val_pow
            else:
                # Partition the range
                len_parts = self.range[1] - self.range[0]
                len_part = len_parts / self.num_parts
                parts = []
                for i in range(0, self.num_parts):
                    part = len_part * i
                    part += self.range[0]
                    parts.append(part)
                val = parts[self.part_index]
        else:
            val = self.range[0]
        self.part_index += 1
        if self.part_index >= self.num_parts:
            self.part_index = 0 
        if self.make_int:
            val = int(val)
        self.val = val
        return self.val

    def pick_from_random(self, search_rng=np.random.default_rng()):
        # Generate random value from the desired range, scaling and openness
        if self.range[0] != self.range[1]:
            if (self.is_log_dist):
                # Generate value from a log distribution
                val_pow_range = np.log10(self.range)
                val_pow = search_rng.random() # Generate uniformly distributed random number from the half-open interval [0, 1)
                if (self.is_rev_open):
                    val_pow = 1 - val_pow # Swap interval openness to (0, 1]
                val_pow *= (val_pow_range[1] - val_pow_range[0]) # Reduce the range
                val_pow += val_pow_range[0] # Displace the range
                val = 10**val_pow
            else:
                # Generate value from a linear distribution
                val = search_rng.random() # Generate random number from the half-open interval [0, 1)
                if (self.is_rev_open):
                    val = 1 - val # Swap interval openness to (0, 1]
                val *= (self.range[1] - self.range[0]) # Reduce the range
                val += self.range[0] # Displace the range
        else:
            # If the range only permits a single value, then just use that value
            val = self.range[0]
        if self.make_int:
            val = int(val)
        self.val = val
        return self.val

def dual_hyperparameter_search(static_hps, variable_hps, data_seed = -1, ann_seed = -1, search_seed = -1, max_iterations = 64, should_make_plots = True, img_type = 'pdf'):
    # Generate (hash) id from static hyperparameters
    static_hps_str = str(list(static_hps.values()))
    static_hps_id = best_hash(static_hps_str)

    # Determine the search_type (Note grid-like not currently supported)
    is_random_search = variable_hps[0].is_random_dist and variable_hps[1].is_random_dist
    search_type = 'random' if is_random_search else 'hybrid'

    # Create local directories for organising text files and figures
    results_dir = f'{variable_hps[0].name}_{variable_hps[1].name}_{search_type}_search_results'
    tu.create_local_dir(results_dir)
    id_dir = f'id-{static_hps_id}'
    tu.create_local_dir(f'{results_dir}/{id_dir}')
    seed_dir = f'data-seed-{data_seed:02d}_ann-seed-{ann_seed:02d}_search-seed-{search_seed:02d}'
    tu.create_local_dir(f'{results_dir}/{id_dir}/{seed_dir}')

    # Create text file
    txt_name = f'{results_dir}/{id_dir}/{seed_dir}/results.txt'
    meta_data = [f'ID: {static_hps_id}', 
                f'Static hyperparameters: ',
                f'? {static_hps}',
                f'Data seed: {data_seed:02d}',
                f'ANN seed: {ann_seed:02d}',
                f'Search seed: {search_seed:02d}'] 
    for hp in variable_hps:
        meta_data.append(f'{hp.name} range: ' + ('(' if hp.is_rev_open else '[') + f'{hp.range[0]}, {hp.range[1]}' + (']' if hp.is_rev_open else ')'))
    # Print the meta data
    print(f'Initialising {variable_hps[0].name} & {variable_hps[1].name} {search_type} search with the following meta-data:')
    for line in meta_data:
        print(line)
    print('')
    column_labels = f'{variable_hps[0].name} ; {variable_hps[1].name} ; final validation loss ; final validation confusion matrix'
    tu.write_hyperparameters(txt_name, meta_data, column_labels)

    # If text file already includes datapoints, then read them
    # -- for continuing crashed runs whilst maintaining deterministic search rng.
    old_hps_to_stats = tu.unload_results(txt_name)
    old_hps = []
    if not isinstance(old_hps_to_stats, type(None)):
        old_hps = list(old_hps_to_stats.keys())

    # Generate the search rng from a fixed seed
    search_rng = ru.generate_rng(search_seed)

    input_dim = None
    if 'patterns' in list(static_hps.keys()):
        # Generate the data rng from a fixed seed
        data_rng = ru.generate_rng(data_seed)

        # Import data
        trn, val = sdg.generate_datasets(static_hps['dataset'],
                                        val_mul=static_hps['val_mul'],
                                        override_patterns=static_hps['patterns'],
                                        try_plot=should_make_plots,
                                        rng=data_rng)

        # Get the input dimension from the training data
        input_dim = len(trn[0][0])

    # Reduce max_iterations if needed, to ensure each partition in grid searches are equally covered
    predicted_incomplete = []
    for hp in variable_hps:
        if not hp.is_random_dist:
            remainder = max_iterations % (hp.num_parts) # Get the remainder (how many over max_iterations)
            incomplete_iterations = remainder # How many under max_iterations # Not sure that this is working...
            if (incomplete_iterations != 0):
                predicted_incomplete.append(incomplete_iterations)
    if predicted_incomplete == []:
        total_incomplete = 0
    else:
        total_incomplete = 1
        for incomplete_iterations in predicted_incomplete:
            total_incomplete *= incomplete_iterations
    max_iterations -= total_incomplete
    i = 0 # Iteration counter
    # Random-search over dropout and l2 for optimal combination
    while i < max_iterations: # Essentially while True... just a bit safer, in case I die and never quit the program -- but then it's not my problem anyway
        hps = {'patterns': None}

        for key in static_hps:
            hps[key] = static_hps[key]
        
        dual_hps = ()
        for hp in variable_hps:
            if hp.is_random_dist:
                hps[hp.name] = hp.pick_from_random(search_rng) 
            else:
                hps[hp.name] = hp.pick_from_grid()
            dual_hps = (*dual_hps, hp.val)

        # Skip pre-existing datapoints (for resuming crashed runs from where they left off)
        # Additionally, ends the run if only a single choice of variable hyperparameters
        if dual_hps in old_hps:
            print(f"These variable hyperparameters already exist! Skipping (i={i}).")
            i += 1
            continue

        # Regenerate the ann rng from the fixed seed
        ann_rng = ru.generate_rng(ann_seed)

        if 'patterns' not in list(static_hps.keys()):
            # Generate the data rng from a fixed seed
            data_rng = ru.generate_rng(data_seed)

            # Import data
            trn, val = sdg.generate_datasets(hps['dataset'],
                                        val_mul=hps['val_mul'],
                                        override_patterns=hps['patterns'],
                                        try_plot=should_make_plots,
                                        rng=data_rng)

            # Get the input dimension from the training data
            input_dim = len(trn[0][0])

        # Properties of all the layers
        # Recipe for defining a layer: [number of nodes, activation function, L2, dropout]
        layer_defines = [[hps['hidden'], act.tanh, hps['l2'], hps['dropout']],
                        [1, act.sig, hps['l2'], hps['dropout']]]
        ann_model = ann.Model(input_dim, layer_defines, ann_rng)

        # Train the network
        ann_model.train(trn, val, hps['lrn_rate'], hps['epochs'], 0, should_save_intermediary_history=should_make_plots)

        if should_make_plots:
            ann_model.show_history(hps['epochs'])

        # Get final validation loss
        fin_val_loss = ann_model.history['val'][-1] 

        # Seperate validation in patterns and targets
        val_patterns = val[0]
        val_targets = val[1]

        # Get final validation accuracy
        val_outputs = ann_model.feed_all_patterns(val_patterns) # Collect the outputs for all the patterns
        val_confusion_matrix = cs.construct_confusion_matrix(val_outputs, val_targets) # Construct confusion matrix

        # Append the final validation loss to the text file
        with open(txt_name, 'a') as f:
            old_hps.append(dual_hps)
            f.write(f'$ {dual_hps[0]} ; {dual_hps[1]} ; {fin_val_loss} ; {val_confusion_matrix.tolist()}\n')

        # Save the error over epochs plot
        if (should_make_plots):
            plot_id = f'{variable_hps[0].name}-{dual_hps[0]}_{variable_hps[1].name}-{dual_hps[1]}'

            # Save the error over epochs plot
            plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/error-over-epochs_{plot_id}.{img_type}') 

            # Plot and save the decision boundary
            if input_dim == 2:
                cs.decision_boundary(val_patterns, val_targets, ann_model)
                plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/validation-decision-boundary_{plot_id}.{img_type}')
            elif input_dim == 1:
                cs.decision_boundary_1d(val_patterns, val_targets, ann_model)
                plt.savefig(f'{results_dir}/{id_dir}/{seed_dir}/validation-decision-boundary_{plot_id}.{img_type}')

        # Increment counter
        i += 1

    plt.show()

    print('Finished!')
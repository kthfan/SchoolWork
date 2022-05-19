
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from scipy.stats import truncnorm
import multiprocessing
from .pso import *
from .utils import *


__all__ = ['HyperParamPSO', 'FocalLossPSO']

np.random.truncnorm = lambda lower, upper, loc, scale, shape:truncnorm.ppf(np.random.uniform(0, 1, shape), lower, upper, loc=loc, scale=scale)

# import time
# class Timer:
#     def __init__(self):
#         self.start()
#     def start(self):
#         self.current_time = time.time()
#         return self.current_time
#     def end(self, label='', to_print=True):
#         s = time.time() - self.current_time
#         if to_print:
#             print(label, s)
#         return s
#     def next(self, label='', to_print=True):
#         s = self.end(label, to_print)
#         self.start()
#         return s
# timer = Timer()


class HyperParamPSO:
    def __init__(self, model_template, num_particles):
        self.num_particles = num_particles
        self.model_template = model_template
        
        self.model_list = [self._copy_model(self.model_template, set_weight=False, compile=False) for i in range(self.num_particles)]
        self.particles = None
        self.pso_solver = PSOSolver(global_tend=[0.5, 0.9])
        self.best_model = None
        self.global_solution = None
        self.particles_boundary = [None, None]
    
    def _copy_model(self, model_template, model=None, set_weight=True, compile=True, **kwds):
        if model is None:
            model = tf.keras.models.clone_model(self.model_template)
        
        if set_weight: 
            model.set_weights(model_template.get_weights())
            
        if model_template.optimizer is not None and compile:
            optimizer = type(model_template.optimizer)(**model_template.optimizer.get_config())
            # set weight of optimizer
            if len(model_template.optimizer.get_weights()) != 0:
                grad_vars = model.trainable_weights
                zero_grads = [tf.zeros_like(w) for w in grad_vars]
                optimizer.apply_gradients(zip(zero_grads, grad_vars))
                optimizer.set_weights(model_template.optimizer.get_weights())
            
            model.compile(optimizer=optimizer, 
                          loss=model_template.loss,
                          metrics=model_template.compiled_metrics._metrics,
                            **kwds)
        return model
            
    def compile(self, optimizer=None, as_model_template=False, loss=None, metrics=None, fitness=None, **kwds):
        if metrics is None:
            metrics = ['accuracy']
        self.fitness = fitness
        self.metrics = metrics
        
        for model in self.model_list:
            if as_model_template:
                self._copy_model(self.model_template, model, set_weight=False, compile=True, **kwds)
            else:
                model.compile(optimizer=type(optimizer)(**optimizer.get_config()), loss=loss, metrics=metrics, **kwds)
    def initialize_param(self):
        raise NotImplementedError('');
        return []
    def param_to_matrix(self, param_list):
        raise NotImplementedError('');
        return np.zero((0, 0))
    def matrix_to_param(self, matrix):
        raise NotImplementedError('');
        return []
    def set_model_param(self, model, param):
        for k, v in param.items():
            if isinstance(v, dict):
                self.set_model_param(model.__dict__[k], v)
            else:
                model.__dict__[k] = v
        
    def charge(self, train_ds, val_ds, validation_steps, particles, charge_iter, pso_workers,
               pso_iter, pso_tol, refresh_weights, pso_patient, k_fold_ds_list=None, **kwds):
        
        # Variables for k fold
        current_k = 0
        k_fold = 0 if k_fold_ds_list is None else len(k_fold_ds_list)
        if k_fold_ds_list is not None:
            train_ds, val_ds = k_fold_ds_list[current_k]
        # Variables for refresh weights
        restore_model = None
        if refresh_weights: restore_model = self._copy_model(self.best_model, set_weight=True, compile=True)
        # Variables for metrics
        metrics = []
        for fn in self.metrics:
            def eval_model(matrix):
                # split val_ds into x, y
                val_x_ds = val_ds.map(lambda x, y: x)
                val_y_ds = val_ds.map(lambda x, y: y)
                val_y = val_y_ds.take(validation_steps).as_numpy_iterator()
                val_y = np.concatenate(list(val_y), axis=0)

                pred_y = self.best_model.predict(val_x_ds)
                return float(fn(val_y, pred_y))
            eval_model.__name__ = fn.__name__
            metrics.append(eval_model)
        self.pso_solver.metrics = metrics

        def train_single_model(model, param):
            # initialize model
            if refresh_weights:
                self._copy_model(restore_model, model, set_weight=True, compile=True)
            self.set_model_param(model, param)
            model.fit(train_ds, epochs=charge_iter, verbose=0, **kwds)
            
            # split val_ds into x, y
            val_x_ds = val_ds.map(lambda x, y: x)
            val_y_ds = val_ds.map(lambda x, y: y)
            val_y = val_y_ds.take(validation_steps).as_numpy_iterator()
            val_y = np.concatenate(list(val_y), axis=0)
                
            pred_y = model.predict(val_x_ds)
            return self.fitness(val_y, pred_y)
        def train_models(params):
            params = self.matrix_to_param(params)
            fitness_score = None
            ## check for workers
            if pso_workers == 1:
                fitness_score = [train_single_model(model, param) for model, param in zip(self.model_list, params)]
            else:
                with multiprocessing.pool.ThreadPool(min(multiprocessing.cpu_count(), self.num_particles, pso_workers)) as pool:
                    fitness_score = pool.map(lambda arg:train_single_model(*arg), [(model, param) for model, param in zip(self.model_list, params)])
                pool.join()

            fitness_score = np.array(fitness_score, dtype=np.float32).reshape((self.num_particles, 1))
            return fitness_score
        
        self.pso_solver.fitness_func = train_models
            
        def each_iter(current_iter, particles):
            nonlocal current_k

            if k_fold_ds_list is not None:
                current_k = (current_k + 1) % k_fold
                train_ds, val_ds = k_fold_ds_list[current_k]
            return False
        
        # run first iteration
        print('initialize models... (it might takes for a while)')
        particles.initialize_variables(fitness_func=train_models)
        best_index = np.argmax(self.particles.best_fitness, axis=0)[0]
        best_param = self.matrix_to_param(np.expand_dims(self.global_solution, 0))[0]
        self.best_model = self._copy_model(self.model_list[best_index], self.best_model, set_weight=True, compile=True)

        # start charging
        result = self.pso_solver.fit(particles, max_iter=pso_iter, tol=pso_tol, 
                        patient=pso_patient, stop_condition=each_iter, verbose=1)
        
        return result

    def sprint(self, model, train_ds, sprint_iter, **kwds):
        history_list = {}
        with tqdm(total=sprint_iter) as pbar:
            for i in range(sprint_iter):
                history = model.fit(train_ds, epochs=1, verbose=0, **kwds)
                pbar.update(1)
                history = dict([(k, v[0]) for k, v in history.history.items()])
                pbar.set_postfix(**history)
                # append history
                for k, v in history.items():
                    history_list[k] = [*history_list.get(k, []), v]
        return history_list
            
    def fit(self, x=None, y=None, batch_size=32, validation_split=0.2, validation_data=None, steps_per_epoch=None, 
            validation_steps=None, validation_batch_size=None, k_fold=None, num_phases=10, charge_iter=20, pso_workers=1,
            pso_iter=10, refresh_weights=True, pso_tol=1e-3, pso_patient=0, sprint_iter=100, **kwds):
        
        ####################### setting arguments #####################
        # check train data is array or dataset
        if isinstance(x, tf.data.Dataset):
            dataset = x
        else:
            if k_fold:
                skf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=True)
                idx = np.concatenate([i for _, i in skf.split(x, np.argmax(y, axis=1))], axis=0)
                x, y = x[idx], y[idx]
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset.batch(batch_size)
        
        # If steps_per_epoch is not set, set to the size according to dataset size and validation_split
        if steps_per_epoch is None:
                steps_per_epoch = int(len(dataset)*(1-validation_split)) if validation_data is None else len(dataset)
        train_ds = dataset.take(steps_per_epoch)
        
        if validation_batch_size is None: validation_batch_size = batch_size
        
        # In our approach (pso search), validation_data is necessary, 
        # hence if not given, we shall take from train data.
        if validation_data is None:
            if validation_steps is None:
                validation_steps = int(steps_per_epoch/(1-validation_split)*validation_split)
            val_ds = dataset\
                    .skip(steps_per_epoch)\
                    .take(validation_steps)
        else:
            if isinstance(validation_data, tuple):
                validation_data = tf.data.Dataset.from_tensor_slices(validation_data).batch(validation_batch_size)
                if validation_steps is None:
                    validation_steps = len(validation_data)
            val_ds = validation_data
        
        # Prepare k fold
        k_fold_ds_list, k_fold_ds = None, None
        if k_fold is not None:
            k_fold_ds = train_ds.concatenate(val_ds)
            chunk_size, remain = steps_per_epoch // k_fold, steps_per_epoch % k_fold
            k_fold_steps = [chunk_size+1 if i < remain else chunk_size for i in range(k_fold)]
            k_fold_offsets = np.concatenate([np.zeros(1), np.cumsum(k_fold_steps)])
            k_fold_ds_list = [(k_fold_ds.take(k_fold_offsets[i]).concatenate(k_fold_ds.skip(k_fold_offsets[i+1])), k_fold_ds.skip(k_fold_offsets[i]).take(k_fold_steps[i])) for i in range(len(k_fold_offsets)-1)]
        
        # record best model and best hyperparameters
        if self.best_model is None:
            self.best_model = self._copy_model(self.model_template, set_weight=True, compile=True)
        if self.global_solution is None:
            self.global_solution = self.param_to_matrix(self.initialize_param())[0]
        
        # save the best model
        def on_global_change(particles, next_global_index, **kwds1):
            self._copy_model(self.model_list[next_global_index], self.best_model, set_weight=True, compile=True)
        self.pso_solver.on_global_change = on_global_change
        
        history = {'charge': [], 'sprint': []}
        ############################ start training ###########################
        for phase in range(num_phases):
            print("================ phase {} ================".format(phase))
            
            # copy num_particles models
            for model in self.model_list: model.set_weights(self.best_model.get_weights())
            
            # preserve last global solution
            random_init_param = self.param_to_matrix(self.initialize_param())
            random_init_param[0] = self.global_solution
            
            self.particles = PSOParticles(random_init_param, 
                                lower_bound=self.particles_boundary[0], 
                                upper_bound=self.particles_boundary[1], 
                                fitness_func=None)
            
            print("Charging...")
            charge_res = self.charge(train_ds, val_ds, validation_steps, particles=self.particles, pso_workers=pso_workers,
                        charge_iter=charge_iter, pso_iter=pso_iter, pso_tol=pso_tol, pso_patient=pso_patient,
                        refresh_weights=refresh_weights, k_fold_ds_list=k_fold_ds_list, **kwds)
            history['charge'].append(charge_res)
            
            # prepare best param and best model
            best_index = np.argmax(self.particles.best_fitness, axis=0)[0]
            self.global_solution = self.particles.global_solution
            best_param = self.matrix_to_param(np.expand_dims(self.global_solution, 0))[0]
            # if self.best_model.optimizer is None:
            #     self.best_model = self._copy_model(self.model_list[best_index], self.best_model, set_weight=True, compile=True)
            # on_global_change
            
            self.set_model_param(self.best_model, best_param)
            
            print("Sprinting...")
            if k_fold is  None:
                sprint_res = self.sprint(self.best_model, train_ds, sprint_iter, **kwds)
                history['sprint'].append(sprint_res)
            else:
                sprint_res = self.sprint(self.best_model, k_fold_ds, sprint_iter, **kwds)
                history['sprint'].append(sprint_res)
        
        return {'model': self.best_model,
                'solution': self.global_solution,
                'history': history}
        
class FocalLossPSO(HyperParamPSO):
    def __init__(self, num_classes, label_num=None, alpha_scale=0.1, gamma_scale=1, **kwds):
        super(FocalLossPSO, self).__init__(**kwds)
        self.num_classes = num_classes
        self.label_num = np.ones(num_classes, dtype=np.float32)/num_classes if label_num is None else np.array(label_num, dtype=np.float32)
        self.alpha_scale = alpha_scale
        self.gamma_scale = gamma_scale

        self.particles_boundary = [np.zeros(2*self.num_classes, dtype=np.float32),
                                    np.concatenate([np.ones(self.num_classes, dtype=np.float32),
                                                    np.inf*np.ones(self.num_classes, dtype=np.float32)])]
    def compile(self, optimizer, **kwds):
        super(FocalLossPSO, self).compile(optimizer, loss=SparseFocalCrossEntropy(alpha=1, gamma=0), **kwds)
    def initialize_param(self):
        if self.global_solution is None:
            self.global_solution = np.concatenate([1/self.label_num/np.sum(1/self.label_num), np.zeros(self.num_classes)])
        base_params = self.global_solution
        
        # initialize params by truncnorm
        params_mat = np.zeros((2*self.num_classes, self.num_particles-1), dtype=np.float32)
        for i in range(self.num_classes):
            aloc = base_params[i]
            rloc = base_params[i+self.num_classes]
            params_mat[i] = np.random.truncnorm(-aloc/self.alpha_scale, (1-aloc)/self.alpha_scale, aloc, self.alpha_scale, self.num_particles-1)
            params_mat[i+self.num_classes] = np.random.truncnorm(-rloc/self.gamma_scale/2, (10-rloc)/self.gamma_scale/2, rloc, 2*self.gamma_scale, self.num_particles-1)
        
        # set params in dict
        params = [{'alpha': self.global_solution[:self.num_classes], 'gamma': self.global_solution[self.num_classes:]}]
        params_mat = params_mat.T  
        for i in range(self.num_particles-1):
            params.append({
                'alpha': params_mat[i, :self.num_classes],
                'gamma': params_mat[i, self.num_classes:]
            })
        return params
    
    def param_to_matrix(self, params):
        particles_matrix = np.zeros((self.num_particles, 2*self.num_classes), dtype=np.float32)
        for i in range(len(params)):
            entry = params[i]
            particles_matrix[i, :self.num_classes] = entry['alpha']
            particles_matrix[i, self.num_classes:] = entry['gamma']
        return particles_matrix
    def matrix_to_param(self, particles_matrix):
        params = []
        for i in range(particles_matrix.shape[0]):
            params.append({
                'alpha': particles_matrix[i, :self.num_classes],
                'gamma': particles_matrix[i, self.num_classes:]
            })
        return params
    
    def set_model_param(self, model, param):
        model.loss.alpha = param['alpha']
        model.loss.gamma = param['gamma']
        
        


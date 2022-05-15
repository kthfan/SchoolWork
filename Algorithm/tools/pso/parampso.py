
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from .pso import *
from .utils import *

__all__ = ['HyperParamPSO', 'FocalLossPSO']

np.random.truncnorm = lambda lower, upper, loc, scale, shape:truncnorm.ppf(np.random.uniform(0, 1, shape), lower, upper, loc=loc, scale=scale)


class HyperParamPSO:
    def __init__(self, model_template, num_particles):
        self.num_particles = num_particles
        self.model_template = model_template
        
        self.model_list = [self._copy_model(self.model_template, set_weight=False, compile=False) for i in range(self.num_particles)]
        self.particles = None
        self.pso_solver = PSOSolver()
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
                          metrics=model_template.metrics,
                            **kwds)
        return model
            
    def compile(self, optimizer=None, as_model_template=False, loss=None, metrics=None, fitness=None, **kwds):
        if metrics is None:
            metrics = ['accuracy']
        self.fitness = fitness
        
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
        
    def charge(self, train_ds, val_ds, validation_steps, particles, charge_iter, 
               pso_iter, pso_tol, refresh_weights, k_fold_ds_list=None, **kwds):
        
        # Variables for k fold
        current_k = 0
        k_fold = 0 if k_fold_ds_list is None else len(k_fold_ds_list)
        if k_fold_ds_list is not None:
            train_ds, val_ds = k_fold_ds_list[current_k]
        # Variables for refresh weights
        restore_model = None
        if refresh_weights: restore_model = self._copy_model(self.best_model, set_weight=True, compile=True)
        
        def train_models(params):
            params = self.matrix_to_param(params)
            fitness_score = np.zeros((self.num_particles, 1), dtype=np.float32)
            for i, model in enumerate(self.model_list):
                # initialize model
                if refresh_weights:
                    self._copy_model(restore_model, model, set_weight=True, compile=True)
                self.set_model_param(model, params[i])
                model.fit(train_ds, epochs=charge_iter, verbose=0, **kwds)
                
                # split val_ds into x, y
                val_x_ds = val_ds.map(lambda x, y: x)
                val_y_ds = val_ds.map(lambda x, y: y)
                val_y = val_y_ds.take(validation_steps).as_numpy_iterator()
                val_y = np.concatenate(list(val_y), axis=0)
                
                pred_y = model.predict(val_x_ds)
                fitness_score[i, 0] = self.fitness(val_y, pred_y)
            return fitness_score
        
        self.pso_solver.fitness_func = train_models
            
        # show the process of PSO search
        with tqdm(total=pso_iter) as pbar:
            pbar.set_description("PSO")
            def show_pso_process(current_iter, particles):
                nonlocal current_k
                pbar.update(1)
                pbar.set_postfix(fitness=particles.global_fitness[0],
                                max_velocity=(particles.velocities**2).sum(axis=1).max()**0.5)
                
                # update k fold
                if k_fold_ds_list is not None:
                    current_k = (current_k + 1) % k_fold
                    train_ds, val_ds = k_fold_ds_list[current_k]
                return False
            
            self.pso_solver.fit(particles, max_iter=pso_iter, tol=pso_tol, stop_condition=show_pso_process)
    def sprint(self, model, train_ds, sprint_iter, **kwds):
        with tqdm(total=sprint_iter) as pbar:
            for i in range(sprint_iter):
                history = model.fit(train_ds, epochs=1, verbose=0)
                pbar.update(1)
                history = dict([(k, v[0]) for k, v in history.history.items()])
                pbar.set_postfix(**history)
            
    def fit(self, x=None, y=None, batch_size=32, validation_split=0.2, validation_data=None, 
            steps_per_epoch=None, validation_steps=None, validation_batch_size=None, k_fold=None,
            num_phases=10, charge_iter=20, pso_iter=10, refresh_weights=True, pso_tol=1e-3, sprint_iter=100, **kwds):
        
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
            self.best_model = self._copy_model(self.model_template, set_weight=False, compile=False)
        if self.global_solution is None:
            self.global_solution = self.param_to_matrix(self.initialize_param())[0]
        
        # save the best model
        def on_global_change(particles, next_global_index, **kwds1):
            self._copy_model(self.model_list[next_global_index], self.best_model, set_weight=True, compile=True)
        self.pso_solver.on_global_change = on_global_change
        
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
            self.charge(train_ds, val_ds, validation_steps, particles=self.particles,
                        charge_iter=charge_iter, pso_iter=charge_iter, pso_tol=pso_tol,
                        refresh_weights=refresh_weights, k_fold_ds_list=k_fold_ds_list)
            
            # prepare best param and best model
            best_index = np.argmax(self.particles.best_fitness, axis=0)[0]
            self.global_solution = self.particles.global_solution
            best_param = self.matrix_to_param(np.expand_dims(self.global_solution, 0))[0]
            best_model = self.model_list[best_index]
            # on_global_change
            
            self.set_model_param(best_model, best_param)
            
            print("Sprinting...")
            if k_fold is  None:
                self.sprint(best_model, train_ds, sprint_iter, **kwds)
            else:
                self.sprint(best_model, k_fold_ds, sprint_iter, **kwds)
        
        return self.best_model, self.global_solution
        
class FocalLossPSO(HyperParamPSO):
    def __init__(self, num_classes, param_scale=2, **kwds):
        super(FocalLossPSO, self).__init__(**kwds)
        self.num_classes = num_classes
        self.param_scale = param_scale
        
        self.particles_boundary = [np.zeros(2*self.num_classes, dtype=np.float32),
                                    np.concatenate([np.ones(self.num_classes, dtype=np.float32),
                                                    np.inf*np.ones(self.num_classes, dtype=np.float32)])]
    def compile(self, optimizer, **kwds):
        super(FocalLossPSO, self).compile(optimizer, loss=SparseFocalCrossEntropy(alpha=1, gamma=0), **kwds)
    def initialize_param(self):
        params = []
        for i in range(self.num_particles):
            params.append({
                'alpha': np.random.truncnorm(-4/self.param_scale, 4/self.param_scale, 0.5, 0.1*self.param_scale, self.num_classes),
                'gamma': np.random.truncnorm(-1/self.param_scale, 5, 1, self.param_scale, self.num_classes)
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
        
        


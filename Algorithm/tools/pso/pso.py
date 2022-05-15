import numpy as np

__all__ = ['PSOSolver', 'PSOParticles']

class PSOSolver:
    def __init__(self, global_tend=0.4, phi=4.1, fitness_func=None):
        
        self.global_tend = global_tend
        self.phi = phi
        self.c1 = self.phi*(1-self.global_tend)
        self.c2 = self.phi*self.global_tend
        
        # https://ieeexplore.ieee.org/document/870279
        self.K = 2 / np.abs(2-phi-(phi**2-4*phi)**0.5)
        self.fitness_func = fitness_func
    def fit(self, particles, max_iter=1000, tol=1e-6, stop_condition=None):
        if isinstance(particles, np.ndarray):
            particles = PSOParticles(particles, self.fitness_func)
        if not particles.is_initialized:
            particles.initialize_variables(fitness_func=self.fitness_func)
            
        history = {'fitness': [], 'solution':[]}
        for current_iter in range(max_iter):
            history['fitness'].append(particles.global_fitness)
            history['solution'].append(particles.global_solution)
            self._single_iteration(particles, current_iter, max_iter)
            if stop_condition is not None and stop_condition(current_iter, particles):
                break
            if (particles.velocities**2).sum(axis=1).max() < tol:
                break
        
        return history
            
    def _single_iteration(self, particles, current_iter, max_iter):
        # change coef durning iterations
        c1 = self.c1 + (self.c2 - self.c1)*current_iter / max_iter
        c2 = self.c2 + (self.c1 - self.c2)*current_iter / max_iter
#         w = 1 + (0.4/self.K - 1)*current_iter / max_iter
        w = 1
        next_velocities = self.K*(w*particles.velocities +\
                c1*np.random.uniform(0,1,1)*(particles.best_solutions - particles.current_solutions) +\
                c2*np.random.uniform(0,1,1)*(particles.global_solution - particles.current_solutions))
        
        next_solutions = particles.current_solutions + next_velocities
        fitness = self.fitness_func(next_solutions)
        particles.update_solutions(next_solutions, fitness, True)
            
class PSOParticles:
    def __init__(self, particles, upper_bound=None, lower_bound=None, fitness_func=None, on_global_change=None):
        self.current_solutions = particles
        self.num_particles = self.current_solutions.shape[0]
        self.dim = self.current_solutions.shape[1]
        if upper_bound is None: upper_bound = np.inf*np.ones(self.dim, dtype=np.float32)
        if lower_bound is None: lower_bound = -np.inf*np.ones(self.dim, dtype=np.float32)
        self.upper_bound = upper_bound.reshape((1, -1))
        self.lower_bound = lower_bound.reshape((1, -1))
        self.on_global_change = on_global_change
        self.is_initialized = False
        self.initialize_variables(fitness_func)
        
        # variable for checking if global solution is changed
        self._last_global_index = -1
    def initialize_variables(self, fitness_func=None):
        self.velocities = np.zeros(self.current_solutions.shape, dtype=np.float32)
        if fitness_func is not None:
            fitness = fitness_func(self.current_solutions)
            
            # set best individual solutions and fitness
            self.best_solutions = self.current_solutions
            self.best_fitness = fitness
            
            # set best global solutions and fitness
            _index = np.argmax(self.best_fitness)
            self.global_solution = self.best_solutions[_index]
            self.global_fitness = self.best_fitness[_index]
            
            self.is_initialized = True
    
    def update_solutions(self, solutions, fitness, update_velocities=True):
        # check boundary of solutions
        is_exceed_bound = np.argwhere(solutions > self.upper_bound)
        solutions[is_exceed_bound[:,0].ravel(), is_exceed_bound[:,1].ravel()] = self.upper_bound[0, is_exceed_bound[:,1].ravel()]
        is_exceed_bound = np.argwhere(solutions < self.lower_bound)
        solutions[is_exceed_bound[:,0].ravel(), is_exceed_bound[:,1].ravel()] = self.lower_bound[0, is_exceed_bound[:,1].ravel()]
        
        if update_velocities: self.velocities = solutions - self.current_solutions
        
        # set best individual solutions and fitness
        should_update = (fitness > self.best_fitness).ravel()
        self.best_solutions[should_update, :] = solutions[should_update, :]
        self.best_fitness[should_update] = fitness[should_update]
        
        # set best global solutions and fitness
        global_index = np.argmax(self.best_fitness)
        _global_solution = self.best_solutions[global_index]
        _global_fitness = self.best_fitness[global_index]
        if not (global_index == self._last_global_index and\
            self.global_fitness == _global_fitness and\
            (_global_solution == self.best_solutions[global_index]).all()):
            if self.on_global_change is not None:
                self.on_global_change(self,
                                      next_global_index=global_index,
                                      next_global_fitness=_global_fitness,
                                      next_global_solution=_global_solution)
                
            self.global_solution = _global_solution
            self.global_fitness = _global_fitness
            self._last_global_index = global_index
        
        self.current_solutions = solutions
    
# if __name__ == '__main__':
    # def gen_func(coeff):
    #     a, b, c = [c for c in np.split(coeff, 3, axis=1)]
        
    #     return lambda x: a*x**2 + b*x + c

    # x_sample = np.linspace(-3, 3)
    # true_coeff = np.array([[2, 1, 5]])
    # corr_func = gen_func(true_coeff)
    # fitness_func = lambda coeff: -np.mean((gen_func(coeff)(x_sample) - corr_func(x_sample))**2, axis=1)


    # particles = PSOParticles(np.random.uniform(-10, 10, (6, 3)), upper_bound=np.array([5, 5, 5]), lower_bound=np.array([-5, -5, -5]), fitness_func=fitness_func)

    # pso = PSOSolver(fitness_func=fitness_func)
    # history = pso.fit(particles, 1000)

    # plt.plot(history['fitness'])


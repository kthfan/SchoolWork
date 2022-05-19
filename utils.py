
import numpy as np
import tensorflow as tf

__all__ = ['SparseFocalCrossEntropy', 
            'WeightedAUC', 
            'WeightedF1Score']

class SparseFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha, gamma, epsilon=1e-6, **kwds):
        super(SparseFocalCrossEntropy, self).__init__(**kwds)
        if not isinstance(alpha, tf.Tensor):
            alpha = tf.constant(alpha, dtype=tf.float32)
        if not isinstance(gamma, tf.Tensor):
            gamma = tf.constant(gamma, dtype=tf.float32)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def get_config(self):
        config = super(SparseFocalCrossEntropy, self).get_config()
        return {
            **config,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
    
    def call(self, y_true, y_pred):
        cross_entropy = -tf.cast(y_true, tf.float32) * tf.math.log(y_pred + self.epsilon)
        loss = self.alpha * tf.math.pow(1 + self.epsilon - y_pred, self.gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

class WeightedAUC:
    def __init__(self, label_weights, steps=200, name='weighted_auc', mode='numpy'):
        self.label_weights = np.array(label_weights, dtype=np.float32)
        self.label_weights = self.label_weights / self.label_weights.sum()
        self.label_weights = self.label_weights.ravel()
        self.steps = steps
        self.mode = mode
        self.name = name
        self.__name__ = name
        if mode == 'tensorflow':
            self.label_weights = tf.constant(self.label_weights, dtype=tf.float32)
    def numpy_call(self, true_y, pred_y):
        pred_y = np.expand_dims(pred_y, axis=-1) > np.linspace(1, 0, self.steps).reshape((1, -1))# N, C, steps
        pred_y = np.transpose(pred_y, (2, 0, 1)) # steps, N, C
        true_y = np.array(true_y).astype(bool)
        
        TP = np.count_nonzero(true_y & pred_y, axis=1)
        FP = np.count_nonzero((~true_y) & pred_y, axis=1)
        P = np.count_nonzero(true_y, axis=0)
        N = true_y.shape[0] - P
        
        TPR = TP / P
        FPR = FP / N
        
        auc = np.sum(np.diff(FPR, axis=0) * (np.diff(TPR, axis=0)/2 + TPR[0:-1, :]), axis=0)
        return np.dot(auc, self.label_weights)
    def tensorflow_call(self, true_y, pred_y):
        pred_y = tf.cast(pred_y, tf.float32)
        pred_y = tf.expand_dims(pred_y, axis=-1) > tf.cast(tf.reshape(tf.linspace(1, 0, self.steps), (1, -1)), tf.float32)# N, C, steps
        pred_y = tf.transpose(pred_y, (2, 0, 1)) # steps, N, C
        true_y = tf.cast(true_y, tf.bool)
        
        TP = tf.math.count_nonzero(true_y & pred_y, axis=1)
        FP = tf.math.count_nonzero((~true_y) & pred_y, axis=1)
        P = tf.math.count_nonzero(true_y, axis=0)
        N = tf.math.count_nonzero(~true_y, axis=0)
        
        TPR = TP / P
        FPR = FP / N
        auc = tf.reduce_sum((FPR[1:]-FPR[:-1]) * ((TPR[1:]-TPR[:-1])/2 + TPR[0:-1, :]), axis=0)
        auc = tf.cast(auc, tf.float32)
        return tf.reduce_sum(auc*self.label_weights, name=self.name)
    def __call__(self, true_y, pred_y):
        if self.mode == 'tensorflow':
            return self.tensorflow_call(true_y, pred_y)
        else:
            return self.numpy_call(true_y, pred_y)


class WeightedF1Score:
    def __init__(self, label_weights, threshold_weights=None, beta=None, epsilon=1e-6,
                 name='weighted_f1_score', mode='numpy'):
        self.label_weights = np.array(label_weights, dtype=np.float32)
        self.label_weights = self.label_weights / self.label_weights.sum()
        self.label_weights = self.label_weights.ravel()
        if threshold_weights is None: threshold_weights = 1
        self.threshold_weights = np.array(threshold_weights, dtype=np.float32)
        if beta is None: beta = 1
        self.beta = beta
        self.mode = mode
        if mode == 'tensorflow':
            self.label_weights = tf.constant(self.label_weights, dtype=tf.float32)
            self.threshold_weights = tf.constant(self.threshold_weights, dtype=tf.float32)
        self.name = name
        self.epsilon = epsilon
        self.__name__ = name
    def numpy_call(self, true_y, pred_y):
        n_cls = true_y.shape[1]
        pred_y = pred_y*self.threshold_weights
        pred_y = np.argmax(pred_y, axis=1).reshape((-1, 1)) == np.arange(n_cls)
        cm =  true_y.astype(np.int32).T @ pred_y.astype(np.int32)
        pred_num = cm.sum(axis=0)
        true_num = cm.sum(axis=1)
        recall = np.diag(cm) / true_num
        precisioin = np.diag(cm) / (pred_num + self.epsilon)
        f1_score = (1+self.beta**2)*precisioin*recall/(self.beta**2*precisioin + recall + self.epsilon)
        
        return np.dot(f1_score, self.label_weights)
    def tensorflow_call(self, true_y, pred_y):
        n_cls = true_y.shape[1]
        pred_y = pred_y*self.threshold_weights
        pred_y = tf.reshape(tf.argmax(pred_y, axis=1), (-1, 1)) == tf.range(n_cls, dtype=tf.int64)
        cm =  tf.cast(tf.transpose(tf.cast(true_y, tf.int32)) @ tf.cast(pred_y, np.int32), tf.float32)
        pred_num = tf.reduce_sum(cm, axis=0)
        true_num = tf.reduce_sum(cm, axis=1)
        recall = tf.linalg.diag_part(cm) / true_num
        precisioin = tf.linalg.diag_part(cm) / (pred_num + self.epsilon)
        
        f1_score = (1+self.beta**2)*precisioin*recall/(self.beta**2*precisioin + recall + self.epsilon)
        return tf.reduce_sum(f1_score*self.label_weights, name=self.name)
    def __call__(self, true_y, pred_y):
        if self.mode == 'tensorflow':
            return self.tensorflow_call(true_y, pred_y)
        else:
            return self.numpy_call(true_y, pred_y)

class CohenKappa:
    def __init__(self, label_weights=None, threshold_weights=None, epsilon=1e-6,
                 name='cohen_kappa', mode='numpy'):
        if label_weights is None: label_weights = 1
        if threshold_weights is None: threshold_weights = 1
        self.label_weights = np.array(label_weights, dtype=np.float32)
        self.label_weights = self.label_weights / self.label_weights.sum()
        self.label_weights = self.label_weights.ravel()
        self.threshold_weights = np.array(threshold_weights, dtype=np.float32)
        
        self.mode = mode
        if mode == 'tensorflow':
            self.label_weights = tf.constant(self.label_weights, dtype=tf.float32)
            self.threshold_weights = tf.constant(self.threshold_weights, dtype=tf.float32)
        self.name = name
        self.epsilon = epsilon
        self.__name__ = name
    def numpy_call(self, true_y, pred_y):
        n_cls = true_y.shape[1]
        pred_y = pred_y*self.threshold_weights
        pred_y = np.argmax(pred_y, axis=1).reshape((-1, 1)) == np.arange(n_cls)
        cm =  true_y.astype(np.int32).T @ pred_y.astype(np.int32)
        total = cm.sum()
        pred_num = cm.sum(axis=0)
        true_num = cm.sum(axis=1)
        pe = np.sum(pred_num * true_num / total**2)
        po = np.diag(cm).sum() / total

        return (po - pe) / (1 - pe + self.epsilon)

    def tensorflow_call(self, true_y, pred_y):
        n_cls = true_y.shape[1]
        pred_y = pred_y*self.threshold_weights
        pred_y = tf.reshape(tf.argmax(pred_y, axis=1), (-1, 1)) == tf.range(n_cls, dtype=tf.int64)
        cm =  tf.cast(tf.transpose(tf.cast(true_y, tf.int32)) @ tf.cast(pred_y, np.int32), tf.float32)
        total = tf.reduce_sum(cm)
        pred_num = tf.reduce_sum(cm, axis=0)
        true_num = tf.reduce_sum(cm, axis=1)
        pe = tf.reduce_sum(pred_num * true_num / total**2)
        po = tf.reduce_sum(tf.linalg.diag_part(cm)) / total
        
        return (po - pe) / (1 - pe + self.epsilon)
    def __call__(self, true_y, pred_y):
        if self.mode == 'tensorflow':
            return self.tensorflow_call(true_y, pred_y)
        else:
            return self.numpy_call(true_y, pred_y)



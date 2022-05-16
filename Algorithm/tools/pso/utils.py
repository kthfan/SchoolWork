
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
        config = super(SparseFocalCrossEntropy, self).get_config(arg)
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

def WeightedAUC(label_weights, steps=200):
    label_weights = np.array(label_weights)
    label_weights = label_weights / label_weights.sum()
    label_weights = label_weights.ravel()
    def weighted_auc(true_y, pred_y):
        pred_y = np.expand_dims(pred_y, axis=-1) > np.linspace(1, 0, steps).reshape((1, -1))# N, C, steps
        pred_y = np.transpose(pred_y, (2, 0, 1)) # steps, N, C
        true_y = np.array(true_y).astype(bool)
        
        
        TP = np.count_nonzero(true_y & pred_y, axis=1)
        FP = np.count_nonzero((~true_y) & pred_y, axis=1)
        P = np.count_nonzero(true_y, axis=0)
        N = true_y.shape[0] - P
        
        TPR = TP / P
        FPR = FP / N
        
        auc = np.sum(np.diff(FPR, axis=0) * (np.diff(TPR, axis=0)/2 + TPR[0:-1, :]), axis=0)
        
        return np.dot(auc, label_weights)
    return weighted_auc

def WeightedF1Score(label_weights, threshold_weights=None, beta=None, epsilon=1e-6):
    label_weights = np.array(label_weights)
    label_weights = label_weights / label_weights.sum()
    label_weights = label_weights.ravel()
    if threshold_weights is None: threshold_weights = 1
    threshold_weights = np.array(threshold_weights, dtype=np.float32)
    if beta is None: beta = 1
    def weighted_f1_score(true_y, pred_y):
        n_cls = true_y.shape[1]
        pred_y = pred_y*threshold_weights
        pred_y = np.argmax(pred_y, axis=1).reshape((-1, 1)) == np.arange(n_cls)
        cm =  true_y.astype(np.int32).T @ pred_y.astype(np.int32)
        pred_num = cm.sum(axis=0).ravel()
        true_num = cm.sum(axis=1).ravel()
        recall = np.diag(cm) / true_num
        precisioin = np.diag(cm) / (pred_num + epsilon)
        f1_score = (1+beta**2)*precisioin*recall/(beta**2*precisioin + recall + epsilon)
        
        return np.dot(f1_score, label_weights)
    return weighted_f1_score


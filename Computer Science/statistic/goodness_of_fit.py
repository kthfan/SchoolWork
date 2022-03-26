
def anderson(sample, cdf=None, pdf=None):
    n = len(sample)
    y = np.unique(sample)
    if cdf is not None:
        cdf = cdf(y)
    elif pdf is not None:
        cdf = pdf(y)
        cdf = np.cumsum(cdf) / n
    else: raise Exception('cdf and pdf are None')
    cdf[cdf==0] = 1e-16
    cdf[cdf==1] = 1-1e-16
    S = np.dot(np.arange(1, 2*n, 2) - 1, np.log(cdf) + np.log(1 - np.flip(cdf))) / n
    A2 = -n - S
    return A2
def test_stats_model(sample, additional_cdf={}, step=0.05, tol=1e-5, max_iter=1000, verbose=False):
    sig = np.var(sample)**0.5
    mu = np.mean(sample)
    n = len(sample)
    y = np.unique(sample)
    
    base_dict = {'loc':mu, 'scale':sig}
    additional_cdf = dict([[k, [v[0], v[1], base_dict if v[2] is None else v[2]]] for k,v in additional_cdf.items()])
    cdf_dict = {
        'uniform': [scipy.stats.uniform.cdf, [], base_dict],
        'norm': [scipy.stats.norm.cdf, [], base_dict],
        'powernorm': [scipy.stats.powernorm.cdf, [1.], base_dict],
        'lognorm': [scipy.stats.lognorm.cdf, [1.], base_dict],
        'skewnorm': [scipy.stats.powernorm.cdf, [1.], base_dict],
        'expon': [scipy.stats.expon.cdf, [], base_dict],
        'weibull_min': [scipy.stats.weibull_min.cdf, [1.], base_dict],
        'weibull_max': [scipy.stats.weibull_max.cdf, [1.], base_dict],
        'genextreme': [scipy.stats.genextreme.cdf, [1], base_dict],
        'gamma': [scipy.stats.gamma.cdf, [1], base_dict],
        'logistic': [scipy.stats.logistic.cdf, [], base_dict],
        't': [scipy.stats.t.cdf, [1], base_dict],
        'chi2': [scipy.stats.chi2.cdf, [1], base_dict],
        **additional_cdf
    }
    result_dict = dict()
    for name, [fcn, args, kwds] in cdf_dict.items():
        argc = len(args)
        kwargc = len(kwds)
        cdf = fcn(y, *args, **kwds)
        cdf[cdf<=0] = 1e-16; cdf[cdf>=1] = 1-1e-16
        last_score = - n - np.dot(np.arange(1, 2*n, 2) - 1, np.log(cdf) + np.log(1 - np.flip(cdf))) / n
        grads_args = [0.01 if isinstance(a, float) else 1 for a in args]
        grads_kwds = dict([[k,0.01] if isinstance(v, float) else [k,1] for k,v in kwds.items()])
        # gradient descent
        for i in range(max_iter):
            du_list = []
            du_dict = dict()
            has_nan = False
            if verbose:
                print('model: {}, iter: {}, score: {}, args: {}, kwds: {}, grad_args: {}, grad_kwds: {}'.format(name, i, last_score, args, kwds, grads_args, grads_kwds))
            # find gradient of args
            for j in range(argc):
                # no gradient
                if grads_args[j] == 0:
                    du_list.append(0); continue
                _args = args.copy(); _args[j] += grads_args[j];
                cdf = fcn(y, *_args, **kwds)
                cdf[cdf<=0] = 1e-16; cdf[cdf>=1] = 1-1e-16
                u = - n - np.dot(np.arange(1, 2*n, 2) - 1, np.log(cdf) + np.log(1 - np.flip(cdf))) / n
                du = (u - last_score) / grads_args[j] * step # diff
                if np.isnan(du):
                    has_nan = True
                    break
                if isinstance(args[j], int): du = int(np.ceil(du)) # cast to int if necessary
                du_list.append(du)
            # find gradient of kwds
            for k,v in kwds.items():
                # no gradient
                if grads_kwds[k] == 0:
                    du_dict[k] = 0; continue
                _kwds = kwds.copy(); _kwds[k] += grads_kwds[k];
                cdf = fcn(y, *args, **_kwds)
                cdf[cdf<=0] = 1e-16; cdf[cdf>=1] = 1-1e-16
                u = - n - np.dot(np.arange(1, 2*n, 2) - 1, np.log(cdf) + np.log(1 - np.flip(cdf))) / n
                du = (u - last_score) / _kwds[k] * step
                if np.isnan(du):
                    has_nan = True
                    break
                if isinstance(kwds[k], int): du = int(np.ceil(du))
                du_dict[k] = du
            if has_nan: break;
            # update params
            _args = [args[j]-du_list[j] for j in range(argc)]
            _kwds = dict([[k, kwds[k]-du_dict[k]] for k in kwds.keys()])
            # compute next score
            cdf = fcn(y, *_args, **_kwds)
            cdf[cdf<=0] = 1e-16; cdf[cdf>=1] = 1-1e-16
            score = - n - np.dot(np.arange(1, 2*n, 2) - 1, np.log(cdf) + np.log(1 - np.flip(cdf))) / n 
            if np.isnan(score): break
            # check if converage
            if np.abs(score - last_score) < tol:
                last_score = score
                args = _args; kwds = _kwds
                break
            last_score = score
            args = _args; kwds = _kwds
            grads_args = du_list; grads_kwds = du_dict
        result_dict[name] = [last_score, args, kwds]
    return result_dict

def mixture_model_cdf(cdf_list, argc_list, weights=None):
    N = len(cdf_list)
    if weights is None: weights = np.ones(N) / N
    weights_trainable = weights == 'trainable'
    if not weights_trainable:  weights = np.array(weights).reshape((1, -1))
    else : weights = np.zeros([1, N], dtype=np.float64)
    
    def cdf(x, *args, **kwds):
        count = 0
        if weights_trainable: 
            count = N
            weights[0, :] = np.array(args[0:count]).reshape((1, -1))

        y = np.zeros((N, len(x)), dtype=np.float64)
        for i in range(N):
            c = argc_list[i]
            fcn = cdf_list[i]
            y[i, :] = fcn(x, *args[count:count+c], **kwds)
            count += c
        return (weights @ y).ravel()
    return cdf
def mixture_model_pdf(pdf_list, argc_list, weights=None):
    N = len(pdf_list)
    if weights is None: weights = np.ones(N) / N
    weights = np.array(weights).reshape((1, -1))
    def cdf(x, *args, **kwds):
        count = 0
        y = np.zeros((N, len(x)), dtype=np.float64)
        for i in range(N):
            c = argc_list[i]
            fcn = pdf_list[i]
            y[i, :] = fcn(x, *args[count:count+c], **kwds)
            count += c
        return (weights @ y).ravel()
    return cdf

def qqplot(sample, ppf, *args, **kwds):
    y = np.sort(sample)
    x = np.linspace(0, 1, len(y))
    x = ppf(x, *args, **kwds)
    plt.scatter(x, y)
    plt.plot(x, x, 'r')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

            

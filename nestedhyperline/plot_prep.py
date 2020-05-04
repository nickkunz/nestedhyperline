## load libraries
import matplotlib.lines as ln
import matplotlib.cm as cm
import numpy as np

## plot pre-processor
def plot_prep(k_outer, n_evals, standardize, results):

    """ Pre-processes the regression results for all plots. """

    ## assign color map
    if k_outer <= 10:
        name = 'tab10'
    if k_outer > 10:
        name = 'tab20'
    if k_outer > 20:
        raise ValueError(
            'Outer k-fold exceeds 20: plot will not display correctly'
            )
    
    ## create color map
    colors = cm.get_cmap(name).colors

    ## lambda, error, coefficient lists
    lamb_list = []
    loss_list = []
    coef_list = []
    leg_lines = []
    leg_labels = []
    lamb_list_test = []

    for i in range(k_outer):
        i_lamb = [i]
        i_loss = [i]
        i_coef = [i]

        for j in range(n_evals):
            i_lamb.append(results.trials_list[i].trials[j]
                ['misc']['vals']['alpha'][0])
            i_loss.append(results.trials_list[i].trials[j]
                ['result']['loss'])
            i_coef.append(results.trials_list[i].trials[j]
                ['result']['coef'])
        
        lamb_list.append(i_lamb)
        loss_list.append(i_loss)
        coef_list.append(i_coef)
        
        lamb_list[i].pop(0)
        loss_list[i].pop(0)
        coef_list[i].pop(0)

        ## lambda test list
        lamb_list_test.append(
            results.params[i].get('alpha')
        )

        ## legend labels
        leg_lines.append(ln.Line2D(
            xdata = [0],
            ydata = [0],
            linewidth = 3,
            color = colors[i]
            )
        )

        leg_labels.append(
            "Outer K-Fold {k}".format(k = i + 1)
        )

    ## average lambda and trian-valid error
    loss_mean = list(np.mean(a = loss_list, axis = 0))
    lamb_mean = list(np.mean(a = lamb_list, axis = 0))

    ## lambda corresponding to lowest average error
    loss_mean_min = min(loss_mean)
    lamb_mean_min = lamb_mean[loss_mean.index(loss_mean_min)]

    ## lambda min and max
    lamb_min = np.min(lamb_list)
    lamb_max = np.max(lamb_list)

    ## lowest test lambda
    lamb_mean_test = np.mean(lamb_list_test)

    ## elastic-net alpha list
    if len(results.trials_list[i].trials[j]['misc']['vals']) == 2:
        alpha_list = []
        alpha_list_test = []

        for i in range(k_outer):
            i_alpha = [i]

            for j in range(n_evals):
                i_alpha.append(results.trials_list[i].trials[j]
                    ['misc']['vals']['l1_ratio'][0])

            alpha_list.append(i_alpha)
            alpha_list[i].pop(0)

            ## alpha test list
            alpha_list_test.append(
                results.params[i].get('l1_ratio')
            )

        ## average trian-valid alpha
        alpha_mean = list(np.mean(a = alpha_list, axis = 0))

        ## alpha corresponding to lowest average error
        loss_mean_min = min(loss_mean)
        alpha_mean_min = alpha_mean[loss_mean.index(loss_mean_min)]

        ## alpha min and max
        alpha_min = np.min(alpha_list)
        alpha_max = np.max(alpha_list)

        ## lowest test alpha
        alpha_mean_test = np.mean(alpha_list_test)

    ## legend label lines
    leg_lines.extend([
        ln.Line2D(
            xdata = [0],
            ydata = [0],
            linewidth = 1.30,
            color = 'black',
            linestyle = 'dotted'
            ),
        ln.Line2D(
            xdata = [0],
            ydata = [0],
            linewidth = 1.1,
            color = 'black'
            )
        ]
    )

    leg_labels.extend([
        'Min Avg Test Err',
        'Min Avg Train Err'
        ]
    )

    ## pre-process results
    return {
        'colors': colors,
        'lamb_min': lamb_min,
        'lamb_max': lamb_max,
        'lamb_list': lamb_list,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'alpha_list': alpha_list,
        'leg_lines': leg_lines,
        'leg_labels': leg_labels,
        'lamb_mean_min': lamb_mean_min,
        'lamb_mean_test': lamb_mean_test,
        'alpha_mean_min': alpha_mean_min,
        'alpha_mean_test': alpha_mean_test
    }

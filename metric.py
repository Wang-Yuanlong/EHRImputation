import numpy as np

def compute_nRMSE(pred, label, mask):
    '''
    same as 3dmice
    '''
    assert pred.shape == label.shape == mask.shape

    missing_indices = mask==1
    missing_pred = pred[missing_indices]
    missing_label = label[missing_indices]
    missing_rmse = np.sqrt(((missing_pred - missing_label) ** 2).mean())

    init_indices = mask==0
    init_pred = pred[init_indices]
    init_label = label[init_indices]
    init_rmse = np.sqrt(((init_pred - init_label) ** 2).mean())

    metric_list = [missing_rmse, init_rmse]
    for i in range(pred.shape[2]):
        apred = pred[:,:,i]
        alabel = label[:,:, i]
        amask = mask[:,:, i]

        mrmse, irmse = [], []
        for ip in range(len(apred)):
            ipred = apred[ip]
            ilabel = alabel[ip]
            imask = amask[ip]

            x = ilabel[imask>=0]
            if len(x) == 0:
                continue

            minv = ilabel[imask>=0].min()
            maxv = ilabel[imask>=0].max()
            if maxv == minv:
                continue

            init_indices = imask==0
            init_pred = ipred[init_indices]
            init_label = ilabel[init_indices]

            missing_indices = imask==1
            missing_pred = ipred[missing_indices]
            missing_label = ilabel[missing_indices]

            assert len(init_label) + len(missing_label) >= 2

            if len(init_pred) > 0:
                init_rmse = np.sqrt((((init_pred - init_label) / (maxv - minv)) ** 2).mean())
                irmse.append(init_rmse)
            else:
                init_rmse = 0
                irmse.append(init_rmse)

            if len(missing_pred) > 0:
                missing_rmse = np.sqrt((((missing_pred - missing_label)/ (maxv - minv)) ** 2).mean())
                mrmse.append(missing_rmse)
            else:
                missing_rmse = 0
                mrmse.append(missing_rmse)

        metric_list.append(np.mean(mrmse))
        metric_list.append(np.mean(irmse))

    metric_list = np.array(metric_list)


    metric_list[0] = np.mean(metric_list[2:][::2])
    metric_list[1] = np.mean(metric_list[3:][::2])

    return metric_list
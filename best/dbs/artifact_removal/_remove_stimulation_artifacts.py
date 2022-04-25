import torch
import numpy as np
from best.dbs.artifact_removal import configs_ArtifactEraser, models_ArtifactEraser
from best.signal import get_datarate, buffer

def remove_artifacts(x, fs, cuda=3):
    """

    :param x: numpy array with shape[N]
    :param fs: 250 or 500 Hz
    :param cuda: integer denoting id of Cuda to be used, alternatively 'cpu' is accepted as well
    :return:
    """

    model = models_ArtifactEraser[f'RCS_ArtifactEraser_{int(fs)}Hz_64']
    model = model.eval()

    if cuda == 'cpu':
        model = model.cpu()
    else:
        model = model.cuda(cuda)

    mn_x = np.nanmean(x)
    x -= mn_x
    nans = np.isnan(x)

    fs = int(fs)
    shape = x.shape[0]
    x = np.concatenate((np.zeros(1 * fs), x, np.zeros(5 * fs)))

    xb = buffer(x, fs, segm_size=2, overlap=1.0)
    mn = np.nanmean(xb, axis=1)
    mn[np.isnan(mn)] = 0

    std = np.nanstd(xb, axis=1)
    std[np.isnan(std)] = 1

    dr = get_datarate(xb)
    xb[np.isnan(xb)] = 0

    xb -= mn.reshape(-1, 1)  # zscore
    xb /= std.reshape(-1, 1)

    dr_pos = np.where(dr > 0.1)[0]  # only datarate > than 0.1 in 30 minute segment
    xb_proc = xb[dr_pos]
    xb[:] = np.NaN
 
    # idx_end = 1
    # if xb_proc.shape[0]-512 > 1:
    #     idx_end = xb_proc.shape[0]-512
    idx_end = x.shape[0]


    with torch.no_grad():
        for k in np.arange(0, idx_end, 512):
            tmp = xb_proc[k:k + 512]
            bs = tmp.shape[0]
            if bs > 0:
                if cuda == 'cpu':
                    tmp_rec, det = model(torch.tensor(tmp).float().view(bs, 1, -1).cpu())
                else:
                    tmp_rec, det = model(torch.tensor(tmp).float().view(bs, 1, -1).cuda(cuda))

                tmp_rec = tmp_rec.detach().view(tmp_rec.shape[0], tmp_rec.shape[2]).cpu().numpy()
                xb_proc[k:k + bs] = tmp_rec

    xb[dr_pos] = xb_proc
    xb *= std.reshape(-1, 1)
    xb += mn.reshape(-1, 1)  # denormalize

    xrec = xb[:, int(0.5 * fs):int(1.5 * fs)].flatten()[
           int(0.5 * fs):int(0.5 * fs) + shape]  # crop only valid semgments
    xrec[nans] = np.nan
    x = xrec
    x += mn_x
    return x



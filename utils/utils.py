
def get_lr(lrs=[1e-3,1e-4,1e-5], epoch=30):
    step = -1
    seg_len = epoch // len(lrs)
    rem_len = epoch % len(lrs)
    lrs_ex = []
    for i in range(len(lrs)):
        lrs_ex += [lrs[i]] * seg_len
    lrs_ex += [lrs[i]] * rem_len
    while step < epoch - 1:
        step += 1
        yield lrs_ex[step]

def change_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

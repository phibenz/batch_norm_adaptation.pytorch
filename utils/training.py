import torch
import logging

from utils.utils import AverageMeter
from utils.utils import one_hot

def validate(val_loader, model, criterion, use_cuda=True, target_class=None):
    logger = logging.getLogger('logbuch')
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        if target_class is not None:
            if len(target.shape) > 1:
                target = one_hot(torch.ones_like(torch.argmax(target, dim=-1)) * target_class)
            else:
                target = torch.ones_like(target) * target_class
        
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
    logger.info('==Test== Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg))
    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

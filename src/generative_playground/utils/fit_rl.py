import torch
from torch.autograd import Variable
from generative_playground.utils.gpu_utils import to_gpu


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_variable(x):
    if type(x) == tuple:
        return tuple([to_variable(xi) for xi in x])
    elif 'ndarray' in str(type(x)):
        return to_gpu(torch.from_numpy(x))
    elif 'Variable' not in str(type(x)):
        return Variable(x)
    else:
        return x


# The fit function is a generator, so one can call several of these in
# the sequence one desires
def fit_rl(train_gen=None, # an iterable providing training data
           model=None,
           optimizer=None,
           scheduler=None,
           epochs=None,
           loss_fn=None,
           grad_clip=5,
           anchor_model=None,
           anchor_weight=0.0,
           callbacks=[],
           metric_monitor=None,
           checkpointer=None
           ):
    print('setting up fit...')
    print('Number of model parameters:', count_parameters(model))
    model.train()
    # loss_fn.train()

    for epoch in range(epochs):
        print('epoch ', epoch)
        if scheduler is not None:
            scheduler.step()
        for inputs in train_gen:
            outputs = model(inputs)
            loss = loss_fn(outputs)
            nice_params = filter(lambda p: p.requires_grad, model.parameters())
            if anchor_model is not None:
                anchor_params = filter(lambda p: p.requires_grad, anchor_model.parameters())
                mean_diffs = [(p1 - p) * (p1 - p) for p1, p in zip(nice_params, anchor_params)]
                cnt = 0
                running_sum = 0
                for d in mean_diffs:
                    running_sum += torch.sum(d)
                    cnt += d.numel()
                anchor_dist = running_sum / cnt
                loss += anchor_weight * anchor_dist
            # do the fit step
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(nice_params, grad_clip)
            optimizer.step()

            # push the metrics out
            this_loss = loss.data.item()
            # do the checkpoint
            # if checkpointer is not None:
            #     avg_loss = checkpointer(None, model, outputs, loss_fn, loss)
            #
            # if metric_monitor is not None:
            #     metric_monitor(None, model, outputs, loss_fn, loss)

            for callback in callbacks:
                if callback is not None:
                    callback(None, model, outputs, loss_fn, loss)
            yield this_loss

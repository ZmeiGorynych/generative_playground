import torch
from torch.autograd import Variable
from generative_playground.utils.gpu_utils import to_gpu

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_variable(x):
    if type(x)==tuple:
        return tuple([to_variable(xi) for xi in x])
    elif 'ndarray'in str(type(x)):
        return to_gpu(torch.from_numpy(x))
    elif 'Variable' not in str(type(x)):
        return Variable(x)
    else:
        return x

# The fit function is a generator, so one can call several of these in
# the sequence one desires
def fit(train_gen = None,
        valid_gen = None,
        model = None,
        optimizer = None,
        scheduler = None,
        epochs = None,
        loss_fn = None,
        batches_to_valid=9,
        grad_clip = 5,
        metric_monitor = None,
        checkpointer = None
        ):

    print('setting up fit...')
    print('Number of model parameters:', count_parameters(model))

    valid_batches = max(1,int(batches_to_valid*len(valid_gen)/len(train_gen)))
    if 'ReduceLROnPlateau' in str(type(scheduler)):
        step_scheduler_after_val = True

    for epoch in range(epochs):
        print('epoch ', epoch)
        if not step_scheduler_after_val:
            scheduler.step()
        train_iter = train_gen.__iter__()
        valid_iter = valid_gen.__iter__()
        done={True:False,False:False}

        for n in range(len(train_gen) + len(valid_gen)):
            if n%(batches_to_valid + valid_batches) <batches_to_valid:
                train = True
                data_iter = train_iter
                model.train()
                loss_fn.train()
            else:
                with torch.no_grad():
                    train = False
                    data_iter = valid_iter
                    model.eval()
                    loss_fn.eval()

            # get the next pair (inputs, targets)
            try:
                inputs_, targets_ = next(data_iter)
            except StopIteration:
                # make sure we get all data from both iterators
                done[train] = True
                if done[True] and done[False]:
                    break
                else:
                    continue

            inputs = to_variable(inputs_)
            targets = to_variable(targets_)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            this_loss = loss.data.item()
            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    nice_params = filter(lambda p: p.requires_grad, model.parameters())
                    torch.nn.utils.clip_grad_norm(nice_params, grad_clip)
                optimizer.step()
            else:
                avg_loss = checkpointer(this_loss,model)
                if step_scheduler_after_val and avg_loss is not None:
                    scheduler.step(avg_loss)

            if metric_monitor is not None:
                metric_monitor(train, this_loss, loss_fn.metrics if hasattr(loss_fn, 'metrics') else None)
            if train:
                yield this_loss



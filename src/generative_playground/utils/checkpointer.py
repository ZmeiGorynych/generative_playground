import torch

# TODO: add an option of triggering on best reward, rather than best loss
class Checkpointer:
    def __init__(self,
                 valid_batches_to_checkpoint=1,
                 save_path=None,
                 save_always=False,
                 verbose=0
                 ):
        self.valid_batches_to_checkpoint = valid_batches_to_checkpoint
        self.save_path = save_path
        self.save_always = save_always
        self.verbose = verbose
        self.val_count = 0
        self.cum_val_loss = 0
        self.best_valid_loss = float('inf')

    def __call__(self, inputs, model, outputs, loss_fn, loss):#val_loss, model):
        print('calling checkpointer...')
        val_loss = loss.data.item()
        if self.verbose > 0:
            print('this loss:' + str(val_loss))
        self.cum_val_loss += val_loss
        self.val_count += 1
        # after enough validation batches, see if we want to save the weights
        if self.val_count >= self.valid_batches_to_checkpoint:
            valid_loss = self.cum_val_loss / self.val_count
            self.val_count = 0
            self.cum_val_loss = 0
            if valid_loss < self.best_valid_loss or self.save_always:
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    print("we're improving!", self.best_valid_loss)
                # spell_out:
                if self.save_path is not None:
                    torch.save(model.state_dict(), self.save_path)
                    print("successfully saved model")

            return valid_loss
        else:
            return None

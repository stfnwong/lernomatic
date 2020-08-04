""""
TEST_INFERRER
Unit tests for Inferrer module

Stefan Wong 2019
"""

import torch
# unit(s) under test
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.train import cifar_trainer
from lernomatic.infer import inferrer
from test import util


def get_model() -> common.LernomaticModel:
    model = cifar.CIFAR10Net()
    return model

def get_trainer(model:common.LernomaticModel,
                checkpoint_name:str,
                batch_size:int,
                save_every:int) -> cifar_trainer.CIFAR10Trainer:
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        batch_size = batch_size,
        test_batch_size = 1,
        device_id = util.get_device_id(),
        checkpoint_name = checkpoint_name,
        save_every = save_every,
        save_hist = False,
        print_every = 50,
        num_epochs = 4,
        learning_rate = 9e-4
    )

    return trainer



class TestInferrer:
    verbose = True

    def test_save_load(self) -> None:
        infer_test_checkpoint = 'checkpoint/infer_save_load_test.pkl'

        model = get_model()
        trainer = get_trainer(model, None, 64, 0)
        # train the model for a while
        trainer.train()
        # save a training checkpoint to disk and load it into an inferrer
        trainer.save_checkpoint(infer_test_checkpoint)

        infer = inferrer.Inferrer(device_id = util.get_device_id())
        infer.load_model(infer_test_checkpoint)

        infer_model = infer.get_model()
        trainer_model = trainer.get_model()

        # check model parameters
        train_model_params = trainer.model.get_net_state_dict()
        infer_model_params = infer.model.get_net_state_dict()
        print('Comparing models')
        for n, (p1, p2) in enumerate(zip(train_model_params.items(), infer_model_params.items())):
            assert p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(train_model_params.items())))
            assert torch.equal(p1[1], p2[1]) == True
        print('\n ...done')

        # run the forward pass
        test_img, _ = next(iter(trainer.val_loader))
        pred = infer.forward(test_img)
        print('Complete prediction vector (shape: %s)' % (str(pred.shape)))
        print(str(pred))

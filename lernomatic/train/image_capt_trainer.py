"""
IMAGE_CAPT_TRAINER
Trainer object for image captioning

Stefan Wong 2019
"""

import time
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
# other lernomatic modules
from lernomatic.train import trainer
from lernomatic.models import image_caption
from lernomatic.util import util

# debug
from pudb import set_trace; set_trace()

def clip_gradient(optimizer, grad_clip):
    """
    CLIP_GRADIENT
    Clips gradients commputed during backpropagation to avoid gradient explosion.
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# TODO : move to scheduler
def adjust_learning_rate(optimizer, shrink_factor):
    """
    ADJUST_LEARNING_RATE
    Shrinks learning rate by a specified factor
    """

    print('Decaying learning rate....')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor

    print('New learning rate is %f\n' % (optimizer.param_groups[0]['lr']))

def ica_accuracy(scores, targets, k):
    """
    ACCURACY
    Computes top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0-dimension tensor

    return correct_total.item() * (100.0 / batch_size)




class ImageCaptTrainer(trainer.Trainer):
    """
    IMAGECAPTTRAINER
    Trainer object for image captioning experiments
    """
    def __init__(self, encoder, decoder, **kwargs):
        self.encoder = encoder
        self.decoder = decoder
        # Deal with keyword args
        self.dec_lr    = kwargs.pop('dec_lr', 1e-4)
        self.dec_mom   = kwargs.pop('dec_mom', 0.0)
        self.dec_wd    = kwargs.pop('dec_wd', 0.0)
        self.enc_lr    = kwargs.pop('enc_lr', 1e-4)
        self.enc_mom   = kwargs.pop('enc_mon', 0.0)
        self.enc_wd    = kwargs.pop('enc_wd', 0.0)
        self.alpha_c   = kwargs.pop('alpha_c', 1.0)
        self.grad_clip = kwargs.pop('grad_clip', None)
        self.word_map  = kwargs.pop('word_map', None)
        super(ImageCaptTrainer, self).__init__(None, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self._init_optimizer()
        self._send_to_device()


    def __repr__(self):
        return 'ImageCaptTrainer'

    def __str__(self):
        s = []
        s.append('ImageCaptTrainer (%d epochs)\n' % self.num_epochs)
        params = self.get_trainer_params()
        s.append('Trainer parameters :\n')
        for k, v in params.items():
            s.append('\t [%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_optimizer(self):
        # optimizer
        if self.decoder is None or self.encoder is None or self.train_loader is None:
            self.decoder = image_caption.DecoderAtten()
            self.encoder = image_caption.Encoder()
            self.decoder_optim = None
            self.encoder_optim = None
        else:
            self.decoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.decoder.parameters()),
                lr = self.dec_lr,
                #momentum = self.dec_mom
            )
            self.encoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.encoder.parameters()),
                lr = self.enc_lr,
                #momentum = self.enc_mom
            )

    def _send_to_device(self):
        if self.decoder is not None:
            self.decoder = self.decoder.to(self.device)
        if self.encoder is not None:
            self.encoder = self.encoder.to(self.device)

    def get_trainer_params(self):
        return {
            # image caption specific parameters
            'dec_lr'          : self.dec_lr,
            'dec_mom'         : self.dec_mom,
            'dec_wd'          : self.dec_wd,
            'enc_lr'          : self.enc_lr,
            'enc_mom'         : self.enc_mom,
            'enc_wd'          : self.enc_wd,
            'alpha_c'         : self.alpha_c,
            'grad_clip'       : self.grad_clip,
            'word_map'        : self.word_map,
            # training params
            'num_epochs'      : self.num_epochs,
            'batch_size'      : self.batch_size,
            'test_batch_tize' : self.test_batch_size,
            # data params
            'shuffle'         : self.shuffle,
            'num_workers'     : self.num_workers,
            # checkpoint params
            'checkpoint_name' : self.checkpoint_name,
            'checkpoint_dir'  : self.checkpoint_dir,
            'save_every'      : self.save_every,
            # display params
            'print_every'     : self.print_every,
            'verbose'         : self.verbose,
            # other
            'device_id'       : self.device_id,
        }

    def set_trainer_params(self, params):
        self.dec_lr          = params['dec_lr']
        self.dec_mom         = params['dec_mom']
        self.dec_wd          = params['dec_wd']
        self.enc_lr          = params['enc_lr']
        self.enc_mom         = params['enc_mom']
        self.enc_wd          = params['enc_wd']
        self.alpha_c         = params['alpha_c']
        self.grad_clip       = params['grad_clip']
        self.word_map        = params['word_map']
        self.num_epochs      = params['num_epochs']
        self.batch_size      = params['batch_size']
        self.test_batch_size = params['test_batch_size']
        self.shuffle         = params['shuffle']
        self.num_workers     = params['num_workers']
        self.checkpoint_name = params['checkpoint_name']
        self.checkpoint_dir  = params['checkpoint_dir']
        self.save_every      = params['save_every']
        self.print_every     = params['print_every']
        self.verbose         = params['verbose']
        self.device_id       = params['device_id']

    def save_checkpoint(self, fname):
        trainer_params = self.get_trainer_params()
        checkpoint_data = {
            # networks
            'encoder' : self.encoder.state_dict(),
            'decoder' : self.decoder.state_dict(),
            'encoder_params' : self.encoder.get_params(),
            'decoder_params' : self.decoder.get_params(),
            # solvers
            'encoder_optim' : self.encoder_optim.state_dict(),
            'decoder_optim' : self.decoder_optim.state_dict(),
            # loaders?
            # object params
            'trainer_state' : trainer_params,
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname):
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer_state'])
        # set meta data
        self.encoder.set_params(checkpoint['encoder_params'])
        self.decoder.set_params(checkpoint['decoder_params'])
        # load weights from checkpoint
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.decoder_optim = torch.optim.Adam(self.decoder.parameters())
        self.encoder_optim = torch.optim.Adam(self.encoder.parameters())
        self.decoder_optim.load_state_dict(checkpoint['decoder_optim'])
        self.encoder_optim.load_state_dict(checkpoint['encoder_optim'])

        if self.device_map is not None:
            if self.device_map < 0:
                device = torch.device('cpu')
            else:
                device = torch.device('cuda:%d' % self.device_map)
            # transfer decoder optimizer
            if self.decoder_optim is not None:
                for state in self.decoder_optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            # trasfer encoder optimizer
            if self.encoder_optim is not None:
                for state in self.encoder_optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            self.device = self.device_map
        self._init_device()
        self._init_dataloaders()
        self.send_to_device()

    def save_history(self, fname):
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history
        if self.acc_history is not None:
            history['acc_history'] = self.acc_history
            history['acc_iter']    = self.acc_iter

        torch.save(history, fname)

    def load_history(self, fname):
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']
        if 'acc_history' in history:
            self.acc_history       = history['acc_history']
            self.acc_iter          = history['acc_iter']
        else:
            self.acc_iter = 0

    # TODO: change these back to VAL?
    def test_epoch(self):
        """
        TEST_EPOCH
        Evaluate accuracy of classifier
        """
        self.decoder.eval()
        if self.encoder is not None:
            self.encoder.eval()

        batch_time = util.AverageMeter()
        losses     = util.AverageMeter()
        top_5_acc  = util.AverageMeter()

        start = time.time()

        references = list()     # references (true captions) for calculating BLEU-4 score
        hypotheses = list()     # hypotheses (predictions)

        for n, (imgs, caps, caplens, allcaps) in enumerate(self.test_loader):
            # move to device
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)
            if self.encoder is not None:
                imgs = self.encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(imgs, caps, caplens)
            # since we decoded starting with <start> the targets are all words
            # after <start> up until <end>
            targets = caps_sorted[:, 1:]
            # Remove timesteps that we didn't decode at, or are pads
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            # Compute loss
            loss = self.criterion(scores, targets)
            # add doubly-stochastic attention regularization
            loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = train_util.accuracy(scores, targets, 5)
            top_5_acc.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if (n % self.print_every) == 0:
                print('[VAL]   Validation [%d/%d]' % (n, len(self.val_loader)))
                print('[VAL]   Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))
                print('[VAL]   Loss       {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))
                print('[VAL]   Top-5 Acc  {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top_5_acc))

            # References
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {self.word_map['<start>'], self.word_map['<pad>']}], img_caps))
                references.append(img_captions)
            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            # remove pads
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # compute the bleu-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        # dump some info to console
        print('\n LOSS : {loss.avg:.3f}, Top-5 Acc : {top5.avg:.3f}, BLEU-4 : {bleu}\n'.format(
            loss=losses,
            top5=top_5_acc,
            bleu=bleu4)
        )

        return bleu4


    def train_epoch(self):
        """
        TRAIN_EPOCH
        Train the classifier for a single epoch
        """
        self.decoder.train()
        if self.encoder is not None:
            self.encoder.train()

        batch_time = util.AverageMeter()        # forward prop + back prop time
        data_time  = util.AverageMeter()        # data loading time
        losses     = util.AverageMeter()        # loss (per word decoded)
        top_5_acc  = util.AverageMeter()        # top5 accuracy

        start = time.time()
        for n, (imgs, caps, caplens) in enumerate(self.train_loader):
            data_time.update(time.time() - start)
            # move data to device
            imgs    = imgs.to(self.device)
            caps    = caps.to(self.device)
            caplens = caplens.to(self.device)

            # forward pass
            enc_imgs = self.encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder(enc_imgs, caps, caplens)
            # since we decoded stating with <start> the targets are all words after
            # <start> and up to <end>
            targets = caps_sorted[:, 1:]
            # remove timesteps that we didn't decode at or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _  = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # compute loss
            loss = self.criterion(scores, targets)
            # add doubly-stochastic attention regularization
            loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            # backprop
            self.decoder_optim.zero_grad()
            if self.encoder_optim is not None:
                self.encoder_optim.zero_grad()
            loss.backward()

            # clip gradients
            if self.grad_clip is not None:
                train_util.clip_gradient(self.decoder_optim, self.grad_clip)
                if self.encoder_optim is not None:
                    train_util.clip_gradient(self.encoder_optim, self.grad_clip)

            # Update weights
            self.decoder_optim.step()
            if self.encoder_optim is not None:
                self.encoder_optim.step()

            # track metrics
            top5 = ica_accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top_5_acc.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # print status
            if n % self.print_every == 0:
                print('[TRAIN] Epoch : [%d] iter [%d/%d]' % (self.cur_epoch, n, len(self.train_loader)), end=' ')
                print('Data time  {data_time.val:.3f}  ({data_time.avg:.3f})'.format(data_time=data_time), end='  ')
                print('Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))
                print('[TRAIN] Loss       {loss.val:.4f}       ({loss.avg:.4f})'.format(loss=losses))
                print('[TRAIN] Top-5 Acc  {top5.val:.3f}       ({top5.avg:.3f})'.format(top5=top_5_acc))

            # save training checkpoint
            if (self.loss_iter % self.save_every) == 0 and self.loss_iter > 0:
                if self.checkpoint_dir is not None:
                    ck_name = str(self.checkpoint_dir) + '/' + str(self.checkpoint_name) + '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                else:
                    ck_name = str(self.checkpoint_name) + '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s], epoch %d, iter %d' % (str(ck_name), self.cur_epoch, self.loss_iter))
                self.save_checkpoint(ck_name)

            # Save loss history
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1


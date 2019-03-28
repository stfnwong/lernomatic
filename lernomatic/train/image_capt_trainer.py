"""
IMAGE_CAPT_TRAINER
Trainer object for image captioning

Stefan Wong 2019
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
# other lernomatic modules
from lernomatic.train import schedule
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
        self.encoder          :common.LernomaticModel = encoder
        self.decoder          :common.LernomaticModel = decoder
        # Deal with keyword args
        self.dec_lr           : float = kwargs.pop('dec_lr', 1e-4)
        self.dec_mom          : float = kwargs.pop('dec_mom', 0.0)
        self.dec_wd           : float = kwargs.pop('dec_wd', 0.0)
        self.enc_lr           : float = kwargs.pop('enc_lr', 1e-4)
        self.enc_mom          : float = kwargs.pop('enc_mon', 0.0)
        self.enc_wd           : float = kwargs.pop('enc_wd', 0.0)
        self.alpha_c          : float = kwargs.pop('alpha_c', 1.0)
        self.grad_clip        : float = kwargs.pop('grad_clip', None)
        self.word_map         : dict  = kwargs.pop('word_map', None)
        self.dec_lr_scheduler         = kwargs.pop('dec_lr_scheduler', None)
        self.enc_lr_scheduler         = kwargs.pop('endec_lr_scheduler', None)
        super(ImageCaptTrainer, self).__init__(None, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self._init_optimizer()
        self._send_to_device()

    def __repr__(self) -> str:
        return 'ImageCaptTrainer'

    def __str__(self) -> str:
        s = []
        s.append('ImageCaptTrainer (%d epochs)\n' % self.num_epochs)
        params = self.get_trainer_params()
        s.append('Trainer parameters :\n')
        for k, v in params.items():
            s.append('\t [%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_optimizer(self) -> None:
        # TODO : ADAM kwargs are betas, eps, weight_decay
        if self.decoder is None:
            self.decoder_optim = None
        else:
            self.decoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.decoder.get_model_parameters()),
                lr = self.dec_lr,
            )

        if self.encoder is None or (self.encoder.do_fine_tune is False):
            self.encoder_optim = None
        else:
            self.encoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.encoder.get_model_parameters()),
                lr = self.enc_lr,
            )

    def _send_to_device(self) -> None:
        if self.decoder is not None:
            self.decoder.send_to(self.device)
        if self.encoder is not None:
            self.encoder.send_to(self.device)

    def get_trainer_params(self) -> dict:
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
            'test_batch_size' : self.test_batch_size,
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

    def set_trainer_params(self, params:dict) -> None:
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


    # set learning rates for the two optimizers
    def set_learning_rate(self, lr:float, param_zero:bool=True) -> None:
        self.set_dec_learning_rate(lr, param_zero)
        self.set_enc_learning_rate(lr, param_zero)

    def set_dec_learning_rate(self, lr:float, param_zero=True) -> None:
        if self.decoder_optim is None:
            return
        if param_zero:
            self.decoder_optim.param_groups[0]['lr'] = lr
        else:
            for g in self.decoder_optim.param_groups:
                g['lr'] = lr

    def set_enc_learning_rate(self, lr:float, param_zero=True) -> None:
        if self.encoder_optim is None:
            return
        if param_zero:
            self.encoder_optim.param_groups[0]['lr'] = lr
        else:
            for g in self.encoder_optim.param_groups:
                g['lr'] = lr

    # Overload set_lr_scheduler
    def set_lr_scheduler(self, lr_scheduler) -> None:
        self.dec_lr_scheduler = lr_scheduler
        self.enc_lr_scheduler = lr_scheduler

    def set_dec_lr_scheduler(self, lr_scheduler) -> None:
        self.dec_lr_scheduler = lr_scheduler

    def set_enc_lr_scheduler(self, lr_scheduler) -> None:
        self.enc_lr_scheduler = lr_scheduler

    def get_dec_lr_scheduler(self) -> None:
        return self.dec_lr_scheduler

    def get_enc_lr_scheduler(self) -> None:
        return self.enc_lr_scheduler

    # Apply the schedule
    def apply_lr_schedule(self) -> None:
        # encoder side
        if self.enc_lr_scheduler is not None:
            if isinstance(self.enc_lr_scheduler, schedule.TriangularDecayWhenAcc):
                new_lr = self.enc_lr_scheduler.get_lr(self.loss_iter, self.acc_history[self.acc_iter])
            elif isinstance(self.enc_lr_scheduler, schedule.EpochSetScheduler) or isinstance(self.enc_lr_scheduler, schedule.DecayWhenEpoch):
                new_lr = self.enc_lr_scheduler.get_lr(self.cur_epoch)
            elif isinstance(self.enc_lr_scheduler, schedule.DecayWhenAcc):
                new_lr = self.enc_lr_scheduler.get_lr(self.acc_history[self.acc_iter])
            else:
                new_lr = self.enc_lr_scheduler.get_lr(self.loss_iter)
            self.set_enc_learning_rate(new_lr)

        # decoder side
        if self.dec_lr_scheduler is not None:
            if isinstance(self.dec_lr_scheduler, schedule.TriangularDecayWhenAcc):
                new_lr = self.dec_lr_scheduler.get_lr(self.loss_iter, self.acc_history[self.acc_iter])
            elif isinstance(self.dec_lr_scheduler, schedule.EpochSetScheduler) or isinstance(self.dec_lr_scheduler, schedule.DecayWhenEpoch):
                new_lr = self.dec_lr_scheduler.get_lr(self.cur_epoch)
            elif isinstance(self.dec_lr_scheduler, schedule.DecayWhenAcc):
                new_lr = self.dec_lr_scheduler.get_lr(self.acc_history[self.acc_iter])
            else:
                new_lr = self.dec_lr_scheduler.get_lr(self.loss_iter)
            self.set_dec_learning_rate(new_lr)

    # ======== TRAINING ======== #
    def train_epoch(self) -> None:
        """
        Train for one epoch
        """
        self.decoder.set_train()
        if self.encoder is not None:
            self.encoder.set_train()

        # TODO : can add batch time meters here later

        for n, (imgs, caps, caplens) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # forward pass
            imgs = self.encoder.forward(imgs)
            scores, caps_sorted, decode_lengths, alphas,  sort_ind = self.decoder.forward(imgs, caps, caplens)
            # TODO (remove after debug) : cache shapes here
            shape_dict = {'scores' : scores.shape, 'caps_sorted' : caps_sorted.shape}
            # remove the <start> token from the output captions
            targets = caps_sorted[:, 1:]
            shape_dict.update({'targets' : targets.shape})
            # remove timesteps that are pads or that we didn't do any decoding
            # for
            scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True)
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            shape_dict.update(
                {'padded_scores' : scores_packed[0].shape,
                 'padded_targets' : targets_packed[0].shape
                }
            )

            # Try this hack (TODO: why does the shape of scores occasionally
            # vary?)
            if scores_packed[0].shape[0] != targets_packed[0].shape[0]:
                print('batch [%d] shapes (%d decode lengths)' % (n, len(decode_lengths)))
                for k, v in shape_dict.items():
                    print('\t%s : %s' % (str(k), str(v)))

            # compute loss
            loss = self.criterion(scores_packed[0], targets_packed[0])
            # add attention regularization
            loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # backward pass
            self.decoder_optim.zero_grad()
            if self.encoder_optim is not None:
                self.encoder_optim.zero_grad()
            loss.backward()

            # perform gradient clip
            if self.grad_clip is not None:
                clip_gradient(self.decoder_optim, self.grad_clip)
                if self.encoder_optim is not None:
                    clip_gradient(self.encoder_optim, self.grad_clip)

            # update weights
            self.decoder_optim.step()
            if self.encoder_optim is not None:
                self.encoder_optim.step()

            # Display
            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            # do scheduling
            if self.dec_lr_scheduler is not None:
                new_lr = self.apply_lr_schedule()
                self.set_dec_learning_rate(new_lr)
            if self.enc_lr_scheduler is not None:
                new_lr = self.apply_lr_schedule()
                self.set_enc_learning_rate(new_lr)

            # update history
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1
            # save checkpoint
            if self.save_every > 0 and n > 0 and (self.loss_iter % self.save_every == 0):
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)


    def test_epoch(self) -> None:
        """
        Find accuracy on test data
        """
        self.decoder.set_eval()
        if self.encoder is not None:
            self.encoder.set_eval()

        references = list()     # true captions for computing BLEU-4 score
        hypotheses = list()     # predicted captions
        correct = 0

        for n, (imgs, caps, caplens, allcaps) in enumerate(self.test_loader):
            # move data to device
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            if self.encoder is not None:
                imgs = self.encoder.forward(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.decoder.forward(imgs, caps, caplens)
            # get rid of the <start> token
            targets = caps_sorted[:, 1:]
            # prune out extra timesteps
            scores_copy    = scores.clone()
            scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True)
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # compute loss, add attention regularization
            test_loss = self.criterion(scores_packed[0], targets_packed[0])
            test_loss += self.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            # display
            if (n % self.print_every) == 0:
                print('[TEST]  :   Epoch       iteration         Test Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.test_loader), test_loss.item()))

            self.test_loss_history[self.test_loss_iter] = test_loss.item()
            self.test_loss_iter += 1

            # references
            allcaps = allcaps[sort_ind]     # decoder sorts images, we re-order here to match
            for cap_idx in range(allcaps.shape[0]):
                img_caps = allcaps[cap_idx].tolist()
                # remove <start> and <pad> tokens
                img_captions = list(
                    map(lambda c: \
                        [w for w in c if w not in {self.word_map.word_map['<start>'], self.word_map.word_map['<pad>']}], img_caps)
                )

                references.append(img_captions)

            # hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for n, pred in enumerate(preds):
                temp_preds.append(preds[n][0 : decode_lengths[n]])
            preds = temp_preds
            hypotheses.extend(preds)

            if len(references) != len(hypotheses):
                raise ValueError('len(references) <%d> != len(hypotheses) <%d>' %\
                                 (len(references), len(hypotheses))
                )

        # print a random hypotheses
        if self.verbose:
            h_idx = np.random.randint(len(hypotheses))
            print('Generated caption %d / %d' % (h_idx, len(hypotheses)))
            caption_text = [self.word_map.lookup_word(w) for w in hypotheses[h_idx]]
            print(str(caption_text))

        # compute the BLEU-4 score
        bleu4 = corpus_bleu(references, hypotheses)
        avg_test_loss = test_loss / len(self.test_loader)
        self.acc_history[self.acc_iter] = bleu4
        self.acc_iter += 1

        print('[TEST]  : Avg. Test Loss : %.4f, BLEU-4 : (%.4f)' %\
              (avg_test_loss, bleu4)
        )

        # save the best weights
        if bleu4 > self.best_acc:
            self.best_acc = bleu4
            if self.save_every > 0:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)


    def eval(self, beam_size=3) -> None:
        self.decoder.set_eval()
        if self.encoder is not None:
            self.encoder.set_eval()

        references = list()
        hypotheses = list()

        for n, (image, caps, caplens, allcaps) in enumerate(self.test_loader):
            k = beam_size
            image = image.to(self.device)       # (1, 3, 256, 256)
            # do encoding pass
            enc_out = self.encoder.forward(image)
            enc_img_size = enc_out.size(1)
            enc_dim      = enc_out.size(3)

            # flatten encoding
            enc_out = enc_out.view(1, -1, enc_dim)      # (1, num_pixels, enc_dim)
            num_pixels = enc_out.size(1)
            # assume the problem has a batch size of k=beam_size
            enc_out = enc_out.expand(k, num_pixels, enc_dim)
            # tensor to store top k previous word at each step (now they're
            # just <start>)
            k_prev_words = torch.LongTensor([[self.word_map.word_map['<start>']]] *k).to(self.device)       # (k, 1)
            seqs = k_prev_words         # tensor to store top-k sequences
            top_k_scores = torch.zeros(k, 1).to(self.device)            # (k, 1)

            complete_seqs  = list()
            complete_seqs_scores = list()

            h, c = self.decoder.init_hidden_state(enc_out)

            # decode
            step = 1
            max_step = 50
            while step < max_step:

                # TODO : some of these need to  be updated to account for new
                # model arch
                embeddings = self.decoder.embedding(k_prev_words).squeeze(1)        # (s, embed_dim)
                awe, _ = self.decoder.attention(enc_out, h)                         # attention-weighted embeddings
                gate = self.decoder.sigmoid(self.decoder.f_beta(h))                 # gating scalar (s, enc_dim)
                awe = awe * gate
                h, c = self.decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))    # (s, dec_dim)

                scores = self.decoder.fc(h)         # (s, vocab_size)
                scores = F.log_softmax(scores, dim=1)

                # add
                scores = top_k_scores.expand_as(scores) + scores        # (s, vocab_size)
                # for the first step, all k points will have the same scores
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)        # (s)
                else:
                    # unroll and find top scores and their unrolled indicies
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

                # convert unrolled indicies to actual indicies
                prev_word_inds = top_k_words / len(self.word_map)
                next_word_indx = top_k_words & len(self.word_map)
                # add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)        # (s, step+1)
                # find any sequences that didn't complete (ie: have no <end>
                # token
                incomplete_inds = [ind for (ind, next_word) in enumerate(next_word_inds) if next_word != self.word_map.word_map['<end>']]
                complete_inds   = list(set(range(len(next_word_inds))) - set(incomplete_inds))

                # set aside complete sequences
                if len(complete_inds) > 0:
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])

                k -= len(complete_inds)

                # proceed with incomplete sequences
                if k == 0:
                    break
                seqs = seqs[incomplete_inds]
                h = h[prev_word_inds[incomplete_inds]]
                c = c[prev_word_inds[incomplete_inds]]
                enc_out = enc_out [prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

                # break if we fail to converge soon enough
                step += 1
                if step > max_step:
                    break

            n = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[n]

            # references
            img_caps = allcaps[0].tolist()
            img_captions = list(
                map(lambda c:
                    [w for w in c if w not in {
                        self.word_map.word_map['<start>'],
                        self.word_map.word_map['<end>'],
                        self.word_map.word_map['<pad>']
                    }], img_caps
                )
            )
            references.append(img_captions)

            # hypotheses
            hypotheses.append([w for w in seq if w not in {
                self.word_map.word_map['<start>'],
                self.word_map.word_map['<end>'],
                self.word_map.word_map['<pad>']
            }])

            assert len(references) == len(hypotheses)

        # compute the BLEU-4 score
        bleu4 = corpus_bleu(references, hypothese)

        return bleu4

    # CHECKPOINTING AND HISTORY
    # TODO : update this for new model arch
    def save_checkpoint(self, fname: str) -> None:
        trainer_params = self.get_trainer_params()
        checkpoint_data = {
            # networks
            'encoder' : self.encoder.get_params() if self.encoder is not None else None,
            'decoder' : self.decoder.get_params() if self.decoder is not None else None,
            # solvers
            'encoder_optim' : self.encoder_optim.state_dict() if self.encoder_optim is not None else None,
            'decoder_optim' : self.decoder_optim.state_dict() if self.decoder_optim is not None else None,
            # loaders?
            # object params
            'trainer_state' : trainer_params,
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname: str) -> None:
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer_state'])
        # set meta data
        self.encoder.set_params(checkpoint['encoder'])
        self.decoder.set_params(checkpoint['decoder'])
        # load weights from checkpoint
        self._init_optimizer()
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
            history['test_loss_iter']    = self.test_loss_iter
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

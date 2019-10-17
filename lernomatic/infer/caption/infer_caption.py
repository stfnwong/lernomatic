""""
INFER_CAPTION
Do foward pass for caption models

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn.functional as F
from lernomatic.infer import inferrer
from lernomatic.models import common
from lernomatic.models import image_caption
from lernomatic.data.text import word_map


class CaptionInferrer(inferrer.Inferrer):
    """
    CaptionInferrer

    Inferrer module for a caption model.
    """
    def __init__(self, wmap:word_map.WordMap,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.word_map = wmap
        self.encoder = kwargs.pop('encoder', None)
        self.decoder = kwargs.pop('decoder', None)
        self.beam_size = kwargs.pop('beam_size', 1) # TODO : default here should be zero (when that is implemented)
        self.max_steps = kwargs.pop('max_steps', 50)

        super(CaptionInferrer, self).__init__(None, **kwargs)
        self._send_to_device()
        if self.encoder is None:
            self.encoder = common.LernomaticModel()
        if self.decoder is None:
            self.decoder = common.LernomaticModel()

    def __repr__(self) -> str:
        return 'CaptionInferrer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    # TODO : this needs to go
    def gen_caption(self, image:torch.Tensor) -> list:
        image = image.to(self.device)
        image = image.unsqueeze(0)

        enc_img = self.encoder.forward(image)
        enc_img_size = enc_img.size(1)
        enc_dim = enc_img.size(3)

        # flatten
        enc_img = enc_img.view(1, -1, enc_dim)  # shape : (1, num_pixels, enc_dim)
        num_pixels = enc_img.size(1)

        # if the beam size is 0, then we don't perform beam searching here.
        # TODO : implement this after the beam search implementation is
        # complete

        # treat the problem as having a batch size of k
        enc_img = enc_img.expand(self.beam_size, num_pixels, enc_dim)   # shape: (k, num_pixels, enc_dim)
        # Store top-k previous words here
        k_prev_words = torch.LongTensor([[self.word_map.get_start()]] * self.beam_size).to(self.device) # (k, 1)
        # Store top-k sequences here. Initially there are just the <start>
        # token
        seqs = k_prev_words
        # Store top-k sequence scores here. Initially they are all 0
        top_k_scores = torch.zeros(self.beam_size, 1).to(self.device)    # shape: (k,1)
        # Store alphas for top-k sequences.
        seq_alphas = torch.ones(self.beam_size, 1, enc_img_size, enc_img_size).to(self.device)

        step = 1
        #h, c = self.decoder.init_hidden_state(enc_img)
        h = None        # hidden state
        c = None        # cell state

        k = self.beam_size

        # lists to hold complete sequences
        complete_seqs = list()
        complete_seqs_alpha = list()
        complete_seqs_scores = list()

        # TODO : factor out this beam search implementation
        while step < self.max_steps:
            #embeddings = self.decoder.embedding(k_prev_words).squeeze(1)
            #atten_w_embed, alpha = self.decoder.attention(enc_img, h)
            #alpha = alpha.view(-1, enc_img_size, enc_img_size)
            #gate = self.decoder.sigmoid(self.decoder.f_beta(h))
            #atten_w_embed = gate * atten_w_embed

            #h, c = self.decoder.lstm(torch.cat([embeddings, atten_w_embed], dim=1), (h ,c))
            #scores = self.decoder.linear(h)
            #scores = F.log_softmax(scores, dim=1)

            scores, alpha, h, c = self.decoder.forward_step(enc_img, enc_img_size, k_prev_words, h, c)

            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            #top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)

            # convert unrolled indicies to actual indicies of scores
            prev_word_inds = top_k_words / len(self.word_map)       # shape (s,)
            next_word_inds = top_k_words % len(self.word_map)       # shape (s,)

            # add new words to sequences, alphas
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)        # shape (s, step+1)
            seq_alphas = torch.cat([seq_alphas[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

            # record sequences that didn't complete (contain no <end> token)
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != self.word_map.get_end()]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            print('[step %d] : found %d complete inds' % (step, len(complete_inds)))

            # we don't need to beam search on complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seq_alphas[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])

            k = k - len(complete_inds)
            if k == 0:
                break

            seqs = seqs[incomplete_inds]
            seq_alphas = seq_alphas[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            enc_img = enc_img[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            step += 1

        ind = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]

        return seq, alphas

    def load_checkpoint(self, fname:str, **kwargs) -> None:
        load_encoder:bool = kwargs.pop('load_encoder', True)
        load_decoder:bool = kwargs.pop('load_decoder', True)
        encoder_key  = kwargs.pop('encoder_key', 'encoder')
        decoder_key  = kwargs.pop('decoder_key', 'decoder')
        if load_encoder:
            self.encoder = common.LernomaticModel()
        else:
            self.encoder = None

        if load_decoder:
            self.decoder = common.LernomaticModel()
        else:
            self.decoder = None

        checkpoint_data = torch.load(fname, map_location='cpu')

        if load_encoder:
            enc_model_path = checkpoint_data['encoder']['model_import_path']
            imp = importlib.import_module(enc_model_path)
            mod = getattr(imp, checkpoint_data['encoder']['model_name'])
            self.encoder = mod()

        if load_decoder:
            dec_model_path = checkpoint_data['decoder']['model_import_path']
            imp = importlib.import_module(dec_model_path)
            mod = getattr(imp, checkpoint_data['decoder']['model_name'])
            self.decoder = mod()

        encoder_params = dict()
        encoder_params.update({'model_name' : checkpoint_data[encoder_key]['model_name']})
        encoder_params.update({'module_name' : checkpoint_data[encoder_key]['module_name']})
        encoder_params.update({'model_import_path' : checkpoint_data[encoder_key]['model_import_path']})
        encoder_params.update({'module_import_path' : checkpoint_data[encoder_key]['module_import_path']})
        encoder_params.update({'model_state_dict' : checkpoint_data[encoder_key]['model_state_dict']})

        decoder_params = dict()
        decoder_params.update({'model_name' : checkpoint_data[decoder_key]['model_name']})
        decoder_params.update({'module_name' : checkpoint_data[decoder_key]['module_name']})
        decoder_params.update({'model_import_path' : checkpoint_data[decoder_key]['model_import_path']})
        decoder_params.update({'module_import_path' : checkpoint_data[decoder_key]['module_import_path']})
        decoder_params.update({'model_state_dict' : checkpoint_data[decoder_key]['model_state_dict']})

        self.encoder.set_params(checkpoint_data[encoder_key])
        self.decoder.set_params(checkpoint_data[decoder_key])

        self._send_to_device()

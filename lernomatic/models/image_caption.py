"""
IMAGE_CAPTION
Models for Image Captioning

Stefan Wong 2018
"""

import torch
import torchvision
from torch import nn

class AttentionNet(nn.Module):
    def __init__(self, enc_dim=1, dec_dim=1, atten_dim=1):
        """
        ATTENTION NETWORK

        Args:
            enc_dim   - size of encoded image features
            dec_dim   - size of decoder's RNN
            atten_dim - size of the attention network
        """
        super(AttentionNet, self).__init__()
        # save dims for __str__
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.atten_dim = atten_dim
        # layers
        self.enc_att  = nn.Linear(enc_dim, atten_dim)    # transform encoded feature
        self.dec_att  = nn.Linear(dec_dim, atten_dim)    # transform decoder output (hidden state)
        self.full_att = nn.Linear(atten_dim, 1)          # compute values to be softmaxed
        self.relu     = nn.ReLU()
        self.softmax  = nn.Softmax(dim=1)       # softmax to calculate weights

    def __repr__(self):
        return 'AttentionNet-%d' % self.atten_dim

    def __str__(self):
        s = []
        s.append('Attention Network\n')
        s.append('Encoder dim: %d, Decoder dim: %d, Attention dim :%d\n' %\
                 (self.enc_dim, self.dec_dim, self.atten_dim))
        return ''.join(s)

    def forward(self, enc_feature, dec_hidden):
        att1 = self.enc_att(enc_feature)        # shape : (N, num_pixels, atten_dim)
        att2 = self.dec_att(dec_hidden)         # shape : (N, atten_dim)
        att  = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)               # shape : (N, num_pixels)
        # compute the attention weighted encoding
        atten_w_enc = (enc_feature * alpha.unsqueeze(2)).sum(dim=1)     # shape : (N, enc_dim)

        return atten_w_enc, alpha

class DecoderAtten(nn.Module):
    def __init__(self, atten_dim=1, embed_dim=1,
                 dec_dim=1, vocab_size=1,
                 enc_dim=2048, dropout=0.5,
                 **kwargs):
        """
        LSTM Decoder for image captioning

        Args:
            atten_dim  - size of attention network
            embed_dim  - size of embedding layer
            dec_dim    - size of decoder RNN
            vocab_size - size of the vocabulary
            enc_dim    - size of encoded features
            dropout    - the dropout ratio
        """
        super(DecoderAtten, self).__init__()
        # copy params
        self.enc_dim    = enc_dim
        self.dec_dim    = dec_dim
        self.atten_dim  = atten_dim
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        self.dropout    = dropout
        self.verbose    = kwargs.pop('verbose', False)
        self.device_id  = kwargs.pop('device_id', -1)
        # create the actual network
        self._init_network()

    def __repr__(self):
        return 'DecoderAtten-%d' % self.dec_dim

    def _init_device(self):
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _init_network(self):
        # Create an Attention network
        self.atten_net   = AttentionNet(self.enc_dim, self.dec_dim, self.atten_dim)
        # Create internal layers
        self.embedding   = nn.Embedding(self.vocab_size, self.embed_dim)
        self.drop        = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.enc_dim, self.dec_dim, bias=True)     # decoding LSTM cell
        # linear layer to find initial hidden state of LSTM
        self.init_h      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to find initial cell state of LSTM
        self.init_c      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to create sigmoid activated gate
        self.f_beta      = nn.Linear(self.dec_dim, self.enc_dim)
        self.sigmoid     = nn.Sigmoid()
        # linear layer to find scores over vocab
        self.fc          = nn.Linear(self.dec_dim, self.vocab_size)
        self.init_weights()
        self._init_device()
        self.send_to_device()

    def init_weights(self):
        """
        Initialize some parameters with values from the uniform distribution
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def send_to_device(self):
        self.atten_net   = self.atten_net.to(self.device)
        self.embedded    = self.embedding.to(self.device)
        self.drop        = self.drop.to(self.device)
        self.decode_step = self.decode_step.to(self.device)
        self.init_h      = self.init_h.to(self.device)
        self.init_c      = self.init_c.to(self.device)
        self.f_beta      = self.f_beta.to(self.device)
        self.sigmoid     = self.sigmoid.to(self.device)
        self.fc          = self.fc.to(self.device)

    def get_params(self):
        params = dict()
        params['enc_dim'] = self.enc_dim
        params['dec_dim'] = self.dec_dim
        params['embed_dim'] = self.embed_dim
        params['atten_dim'] = self.atten_dim
        params['vocab_size'] = self.vocab_size
        params['dropout'] = self.dropout
        params['verbose'] = self.verbose
        params['device_id'] = self.device_id

        return params

    def set_params(self, params):
        self.enc_dim = params['enc_dim']
        self.dec_dim = params['dec_dim']
        self.embed_dim = params['embed_dim']
        self.atten_dim = params['atten_dim']
        self.vocab_size = params['vocab_size']
        self.dropout = params['dropout']
        self.verbose = params['verbose']
        self.device_id = params['device_id']
        self._init_network()

    def load_pretrained_embeddings(self, embeddings):
        """
        INIT_WEIGHTS
        Init some parameters with values from the uniform distribution.
        This improves convergence
        """
        self.embedding.weights = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, tune=True):
        """
        FINE_TUNE_EMBEDDINGS
        """
        for p in self.embedding.parameters():
            p.requires_grad = tune

    def init_hidden_state(self, enc_feature):
        """
        INIT_HIDDEN_STATE
        Initialize the decoders hidden state
        """
        mean_enc_out = enc_feature.mean(dim=1)
        h = self.init_h(mean_enc_out)
        c = self.init_c(mean_enc_out)

        return h, c

    def forward(self, enc_feature, enc_capt, capt_lengths):
        """
        FOWARD PASS

        Args:
            enc_feature  - Tensor of dimension (N, enc_image_size, enc_image_size, enc_dim)
            enc_capt     - Tensor of dimension (N, max_capt_length)
            capt_lengths - Tensor of dimension (N, 1)

        Return:
            Tuple of
            (scores for vocab, sorted encoded captions, decode lengths, weights, sort indices)

        """
        N = enc_feature.size(0)
        enc_dim = enc_feature.size(-1)
        vocab_size = self.vocab_size

        # flatten image
        enc_feature = enc_feature.view(N, -1, enc_dim)  # aka : (N, num_pixels, enc_dim)
        num_pixels = enc_feature.size(1)
        # Sort input data by decreasing lengths (why? see below)
        capt_lengths, sort_ind = capt_lengths.squeeze(1).sort(dim=0, descending=True)
        enc_feature = enc_feature[sort_ind]
        enc_capt = enc_capt[sort_ind]
        # Embeddings
        embeddings = self.embedding(enc_capt)       # shape = (N, dec_dim)
        # init LSTM state
        h, c = self.init_hidden_state(enc_feature)  # shape = (N, dec_dim)
        # we won't decode the <end? position, since we've finished generating
        # as soon as we generate <end>. Therefore decoding lengths are
        # actual_lengths-1
        decode_lengths = (capt_lengths - 1).tolist()
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(N, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(N, max(decode_lengths), num_pixels).to(self.device)

        # At each time step we decode the by
        # 1) Attention-weighting the encoded features based the on the previous
        #    decoder state
        # 2) Generate a new word in the decoder with the previous word and the
        #    attention-weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            atten_w_enc, alpha = self.atten_net(
                enc_feature[:batch_size_t],
                h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(
                h[:batch_size_t])
            )       # gating scalar (batch_size_t, enc_dim)
            atten_w_enc = atten_w_enc * gate
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], atten_w_enc], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )   # shape is (batch_size_t, vocab_size)

            preds = self.fc(self.drop(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, ] = alpha

        return predictions, enc_capt, decode_lengths, alphas, sort_ind

class Encoder(nn.Module):
    """
    CNN Encoder
    """
    def __init__(self, **kwargs): #feature_size=14, do_fine_tune=True):
        super(Encoder, self).__init__()
        self.enc_img_size = kwargs.pop('feature_size', 14)
        self.do_fine_tune = kwargs.pop('do_fine_tune', True)
        self.device_id    = kwargs.pop('device_id', -1)
        #  get network
        self._init_network()

    def _init_network(self):
        resnet = torchvision.models.resnet101(pretrained=True)
        # remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.net = nn.Sequential(*modules)
        # resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_img_size, self.enc_img_size))

        # set the device
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)
        self.net = self.net.to(self.device)
        self.adaptive_pool = self.adaptive_pool.to(self.device)

        if self.do_fine_tune:
            self.fine_tune()

    def get_params(self):
        params = dict()
        params['enc_img_size'] = self.enc_img_size
        params['do_fine_tune'] = self.do_fine_tune
        params['device_id'] = self.device_id
        return params

    def set_params(self, params):
        self.enc_img_size = params['enc_img_size']
        self.do_fine_tune = params['do_fine_tune']
        self.device_id = params['device_id']
        self._init_network()

    def fine_tune(self, tune=True):
        """
        Allow or prevent the computation of gradients for
        convolutional blocks 2 through 4 of the encoder (resnet 101)
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # if fine-tuning only fine tune convolutional blocks 2 through 4
        for c in list(self.net.children())[5:]:
            for p in c.parameters():
                p.requires_grad = tune

    def forward(self, X):
        """
        Forward propagation
        """
        out = self.net(X)                   # (batch_size, 2048, X.size/32, X.size/32)
        out = self.adaptive_pool(out)       # (batch_size, 2048, enc_img_size, enc_img_size)
        out = out.permute(0, 2, 3, 1)       # (batch_size, enc_img_size, enc_img_size, 2048)_

        return out

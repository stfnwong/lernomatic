"""
IMAGE_CAPTION
Models for Image Captioning

Stefan Wong 2018
"""

import torch
import torchvision
import importlib
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from lernomatic.models import common
from lernomatic.models.attention import linear_atten
# caption utils
from lernomatic.util import caption_util

# debug
#from pudb import set_trace; set_trace()

# TODO : change names here (mostly so that the corresponding __repr__ changes
# as well)


# TODO : decoder without attention
# ======== Decoders ======== #
class DecoderAtten(common.LernomaticModel):
    """
    DecoderAtten

    Caption decoder with attenion.
    """
    def __init__(self,
                 atten_dim: int = 512,
                 embed_dim: int = 512,
                 dec_dim: int = 512,
                 vocab_size: int = 1,
                 enc_dim: int = 2048,
                 dropout: float = 0.5,
                 **kwargs) -> None:
        self.net = DecoderAttenModule(
            atten_dim, embed_dim, dec_dim, vocab_size,
            enc_dim, dropout, **kwargs)
        self.model_name         = 'DecoderAtten'
        self.module_name        = 'DecoderAttenModule'
        self.import_path        = 'lernomatic.models.image_caption.image_caption'
        self.module_import_path = 'lernomatic.models.image_caption.image_caption'

    def __repr__(self) -> str:
        return 'DecoderAtten'

    def __str__(self) -> str:
        return 'DecoderAtten-%d' % self.net.vocab_size

    def send_to(self, device:torch.device) -> None:
        self.net.send_to_device(device)

    def embedding(self, X: torch.Tensor) -> torch.Tensor:
        return self.net.embedding(X)

    def attention(self, X: torch.Tensor, hidden_state) -> torch.Tensor:
        return self.net.atten_net(X, hidden_state)

    def sigmoid(self, X:torch.Tensor) -> torch.Tensor:
        return self.net.sigmoid(X)

    def linear(self, X:torch.Tensor) -> torch.Tensor:
        return self.net.fc(X)

    def forward(self,
                enc_feature:torch.Tensor,
                enc_capt:torch.Tensor,
                capt_lengths:torch.Tensor) -> torch.Tensor:
        return self.net(enc_feature, enc_capt, capt_lengths)

    def forward_step(self,
                     enc_feature:torch.Tensor,
                     enc_feature_size:int,
                     enc_capt:torch.Tensor,
                     h:torch.Tensor=None,
                     c:torch.Tensor=None) -> tuple:
        return self.net.forward_step(enc_feature, enc_feature_size, enc_capt, h, c)

    def init_hidden_state(self, X:torch.Tensor) -> tuple:
        return self.net.init_hidden_state(X)

    def f_beta(self, X:torch.Tensor) -> torch.Tensor:
        return self.net.f_beta(X)

    def decode_step(self, X:torch.Tensor) -> torch.Tensor:
        return self.net.lstm(X)

    def get_params(self) -> dict:
        return {
            'model_state_dict'   : self.net.state_dict(),
            'model_name'         : self.get_model_name(),
            'model_import_path'  : self.get_model_path(),
            'module_name'        : self.get_module_name(),
            'module_import_path' : self.get_module_import_path(),
            'atten_params'       : self.net.get_params(),
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod()
        self.net.set_params(params['atten_params'])
        self.net.load_state_dict(params['model_state_dict'])




# ======== MODULES ======== #
class DecoderAttenModule(nn.Module):
    def __init__(self, atten_dim=1, embed_dim=1,
                 dec_dim=1, vocab_size=1,
                 enc_dim=2048, dropout=0.5,
                 **kwargs) -> None:
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
        super(DecoderAttenModule, self).__init__()
        # copy params
        self.enc_dim    = enc_dim
        self.dec_dim    = dec_dim
        self.atten_dim  = atten_dim
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        self.dropout    = dropout
        self.device     = None      # archive the device for some internal forward pass stuff
        # create the actual network
        self._init_network()
        self.init_weights()

    def __repr__(self) -> str:
        return 'DecoderAtten-%d' % self.dec_dim

    def _init_network(self) -> None:
        # Create an Attention network
        # TODO : pass in a seperate attention network?
        self.atten_net   = linear_atten.AttentionNetModule(self.enc_dim, self.dec_dim, self.atten_dim)
        # Create internal layers
        self.embedding   = nn.Embedding(self.vocab_size, self.embed_dim)
        self.drop        = nn.Dropout(p=self.dropout)
        self.lstm        = nn.LSTMCell(self.embed_dim + self.enc_dim, self.dec_dim, bias=True)     # decoding LSTM cell
        # linear layer to find initial hidden state of LSTM
        self.init_h      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to find initial cell state of LSTM
        self.init_c      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to create sigmoid activated gate
        self.f_beta      = nn.Linear(self.dec_dim, self.enc_dim)
        self.sigmoid     = nn.Sigmoid()
        # linear layer to find scores over vocab
        self.fc          = nn.Linear(self.dec_dim, self.vocab_size)

    def forward(self,
                enc_feature:torch.Tensor,
                enc_capt:torch.Tensor,
                capt_lengths:torch.Tensor) -> tuple:
        """
        FOWARD PASS (training)

        Args:
            enc_feature  - Tensor of dimension (N, enc_image_size, enc_image_size, enc_dim)
            enc_capt     - Tensor of dimension (N, max_capt_length)
            capt_lengths - Tensor of dimension (N, 1)

        Return:
            Tuple of
            (scores for vocab, sorted encoded captions, decode lengths, weights, sort indices)

        """
        N           = enc_feature.size(0)             # batch size
        enc_dim     = enc_feature.size(-1)
        vocab_size  = self.vocab_size
        # flatten image
        enc_feature = enc_feature.view(N, -1, enc_dim)  # shape : (N, num_pixels, enc_dim)
        num_pixels  = enc_feature.size(1)

        # pack_padded_sequence expects a sorted tensor (ie: longest to
        # shortest)
        capt_lengths, sort_ind = capt_lengths.squeeze(1).sort(dim=0, descending=True)
        enc_feature = enc_feature[sort_ind]
        enc_capt    = enc_capt[sort_ind]

        # Embeddings
        embeddings = self.embedding(enc_capt)       # shape = (N, max_capt_len, embed_dim)
        # init LSTM state
        h, c = self.init_hidden_state(enc_feature)  # shape = (N, dec_dim)
        # we won't decode the <end> position, since we've finished generating
        # as soon as we generate <end>. Therefore decoding lengths are
        # actual_lengths-1
        decode_lengths = (capt_lengths - 1).tolist()
        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(N, max(decode_lengths), vocab_size).to(self.device)
        alphas      = torch.zeros(N, max(decode_lengths), num_pixels).to(self.device)

        # TODO : Check if we are in train or eval mode here. If we are in train
        # mode then run the loop where we manually apply the attention network
        # to each step of the RNN. If we are in eval mode then just run a step

        # At each time step we decode the sequence by
        # 1) Attention-weighting the encoded features based the on the previous
        #    decoder state
        # 2) Generate a new word in the decoder with the previous word and the
        #    attention-weighted encoding

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            atten_w_enc, alpha = self.atten_net(
                enc_feature[0 : batch_size_t],
                h[0 : batch_size_t]
            )
            pred, h, c = self.lstm_step(
                batch_size_t,
                atten_w_enc,
                embeddings,
                h, c,
                t
            )

            predictions[0 : batch_size_t, t, :] = pred
            alphas[0 : batch_size_t, t, ]       = alpha

        #step_info = {
        #    'enc_capt'       : enc_capt,
        #    'decode_lengths' : decode_lengths, '
        #    'sort_ind'       : sort_ind
        #}

        #return (predictions, alphas, step_info)
        return (predictions, enc_capt, decode_lengths, alphas, sort_ind)

    # TODO : move attention network out of this loop. Since this is an image
    # caption problem we only need the encoded image once (which may not be
    # true for a video caption problem).
    def lstm_step(self,
                  batch_size:int,
                  atten_w_enc:torch.Tensor,
                  embeddings:torch.Tensor,
                  hidden:torch.Tensor,
                  cell:torch.Tensor,
                  t:int) -> tuple:
        """
        Do one step of LSTM
        """

        # Get the attention weighting
        # concat the embeddings with the attention-weighted encodings and
        # return the hidden and cell states (Actually this comment is a bit
        # redundant, better to explain the actual rationale behind this
        # implementation)
        h, c = self.lstm(
            torch.cat([embeddings[0 : batch_size, t, :], atten_w_enc], dim=1),
            (hidden[0 : batch_size], cell[0 : batch_size])          # shape : (batch_size, vocab_size)
        )
        pred = self.fc(self.drop(h))

        return (pred, h, c)

    def forward_step(self,
                     enc_feature:torch.Tensor,
                     enc_feature_size:int,
                     enc_caption:torch.Tensor,
                     h:torch.Tensor=None,
                     c:torch.Tensor=None
                     ) -> tuple:
        """
        FORWARD PASS (evaluation)

        Performs one step of the forward pass
        """
        if h is None or c is None:
            h, c = self.init_hidden_state(enc_feature)

        embeddings = self.embedding(enc_caption)
        atten_w_embed, alpha = self.atten_net(enc_feature, h)
        alpha = alpha.view(-1, enc_feature_size, enc_feature_size)

        gate = self.sigmoid(self.f_beta(h))
        atten_w_embed = gate * atten_w_embed

        scores = self.fc(h)
        scores = F.log_softmax(scores, dim=1)

        return (scores, alpha, h, c)


    def get_params(self) -> dict:
        params = dict()
        params['enc_dim']    = self.enc_dim
        params['dec_dim']    = self.dec_dim
        params['embed_dim']  = self.embed_dim
        params['atten_dim']  = self.atten_dim
        params['vocab_size'] = self.vocab_size
        params['dropout']    = self.dropout
        params['atten_net_dict'] = self.atten_net.state_dict()
        params['atten_net_params'] = self.atten_net.get_params()

        return params

    def set_params(self, params: dict) -> None:
        self.enc_dim = params['enc_dim']
        self.dec_dim = params['dec_dim']
        self.embed_dim = params['embed_dim']
        self.atten_dim = params['atten_dim']
        self.vocab_size = params['vocab_size']
        self.dropout = params['dropout']
        self._init_network()
        # load the attention network parameters
        self.atten_net.set_params(params['atten_net_params'])
        self.atten_net.load_state_dict(params['atten_net_dict'])

    def init_weights(self) -> None:
        """
        INIT_WEIGHTS
        Initialize some parameters with values from the uniform distribution
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def send_to_device(self, device:torch.device) -> None:
        self.atten_net   = self.atten_net.to(device)
        self.embedding   = self.embedding.to(device)
        self.drop        = self.drop.to(device)
        self.lstm        = self.lstm.to(device)
        self.init_h      = self.init_h.to(device)
        self.init_c      = self.init_c.to(device)
        self.f_beta      = self.f_beta.to(device)
        self.sigmoid     = self.sigmoid.to(device)
        self.fc          = self.fc.to(device)
        # save a reference to the device
        self.device      = device

    def load_pretrained_embeddings(self, embeddings) -> None:
        self.embedding.weights = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, tune=True) -> None:
        for p in self.embedding.parameters():
            p.requires_grad = tune

    def init_hidden_state(self, enc_feature) -> tuple:
        """
        INIT_HIDDEN_STATE
        Initialize the decoders hidden state
        """
        mean_enc_out = enc_feature.mean(dim=1)
        # Note that these are (Linear) layers  and not methods or attributes
        h = self.init_h(mean_enc_out)
        c = self.init_c(mean_enc_out)

        return (h, c)



# ======== Encoders ======== #
class Encoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = EncoderModule(**kwargs)
        self.model_name = 'Encoder'
        self.module_name = 'EncoderModule'
        self.import_path = 'lernomatic.models.image_caption.image_caption'
        self.module_import_path = 'lernomatic.models.image_caption.image_caption'

    def __repr__(self) -> str:
        return 'Encoder'

    def send_to(self, device:torch.device) -> None:
        self.net.send_to(device)

    def do_fine_tune(self) -> bool:
        return self.net.do_fine_tune

    def set_fine_tune(self) -> None:
        self.net.fine_tune(True)

    def unset_fine_tune(self) -> None:
        self.net.fine_tune(False)

    def get_params(self) -> dict:
        params = {
            'model_state_dict'   : self.net.state_dict(),
            'model_name'         : self.get_model_name(),
            'model_import_path'  : self.get_model_path(),
            'module_name'        : self.get_module_name(),
            'module_import_path' : self.get_module_import_path(),
            'enc_params'       : self.net.get_params()
        }
        return params

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod()
        self.net.set_params(params['enc_params'])
        self.net.load_state_dict(params['model_state_dict'])



class EncoderModule(nn.Module):
    """
    CNN Encoder for image captioning
    """
    def __init__(self, **kwargs) -> None: #feature_size=14, do_fine_tune=True):
        super(EncoderModule, self).__init__()
        self.enc_img_size = kwargs.pop('enc_img_size', 14)
        self.do_fine_tune = kwargs.pop('do_fine_tune', True)
        #  get network
        self._init_network()

    def _init_network(self) -> None:
        resnet = torchvision.models.resnet101(pretrained=True)
        # remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.net = nn.Sequential(*modules)
        # resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_img_size, self.enc_img_size))
        self.fine_tune(self.do_fine_tune)

    def send_to(self, device:torch.device) -> None:
        self.net = self.net.to(device)
        self.adaptive_pool = self.adaptive_pool.to(device)
        self.device = device

    def get_params(self) -> dict:
        params = dict()
        params['enc_img_size'] = self.enc_img_size
        params['do_fine_tune'] = self.do_fine_tune
        return params

    def set_params(self, params:dict) -> None:
        self.enc_img_size = params['enc_img_size']
        self.do_fine_tune = params['do_fine_tune']
        self._init_network()

    def forward(self, X) -> torch.Tensor:
        """
        Forward propagation
        """
        out = self.net(X)                   # (batch_size, 2048, X.size/32, X.size/32)
        out = self.adaptive_pool(out)       # (batch_size, 2048, enc_img_size, enc_img_size)
        out = out.permute(0, 2, 3, 1)       # (batch_size, enc_img_size, enc_img_size, 2048)_

        return out

    def fine_tune(self, tune=True) -> None:
        """
        Allow or prevent the computation of gradients for
        convolutional blocks 2 through 4 of the encoder (resnet 101)
        """
        self.do_fine_tune = tune
        for p in self.net.parameters():
            p.requires_grad = False
        # if fine-tuning only fine tune convolutional blocks 2 through 4
        if self.do_fine_tune:
            for c in list(self.net.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = True

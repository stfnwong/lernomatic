"""
MNIST_VAE
Variational Autoencoder for use with MNIST dataset

Stefan Wong 2019
"""
from torch import nn
from torch.autograd import Variable

# debug
#from pudb import set_trace; set_trace()

class MNISTVAE(nn.Module):
    def __init__(self, zdims, **kwargs):
        super(MNISTVAE, self).__init__()
        self.zdims = zdims
        self.input_size = kwargs.pop('input_size', 784)
        self.hidden_size = kwargs.pop('hidden_size', 400)
        # Encoder side
        # 28x28 pixels = 784 input pixels, 400 outputs
        self.fc1  = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(self.hidden_size, self.zdims)       # mu layer
        self.fc22 = nn.Linear(self.hidden_size, self.zdims)       # logvariance layer
        self.fc3 = nn.Linear(self.zdims, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, self.input_size)
        self.sigmoid = nn.Sigmoid()

    def encode(self, X):
        """
        ENCODE

        Args:
            X - Input vector. Contains digit vector of 128 digits, each
                28x28 pixels. (Shape : (128, 784)

        Returns:
            mu     - Mean units, one for each latent dimension
            logvar - Log variance, one for each latent dimension
        """
        # h1 is (128, 400)
        h1 = self.relu(self.fc1(X))     # type is Torch.Variable
        return (self.fc21(h1), self.fc22(h1))

    def reparam(self, mu, logvar):
        """
        REPARAM
        Reparameterizes a training sample

        The idea here is:

            - Take the current learned (mu, std) for each of the self.zdims
              dimensions and draw a random sample from that distribution
            - Train the network such that these randomly drawn samples decode
              to output that is similar to the input.
            - This means that the (mu, std) will be learned distributions that
              correctly encode the inputs.
            - Because of the additional KL Divergence term the distribution
              will tend towards unit Gaussians
        """

        if self.training:
            # multiply log variance with 0.5, then do in-place exponend to
            # yeild the standard deviation
            std = logvar.mul(0.5).exp_()        # torch.Variable
            """
            std.data is a (128, self.zdims) tensor. Therefore, eps will be a (128, self.zdims)
            tensor draw from a distribution with mean = 0.
            """
            eps = Variable(std.data.new(std.size()).normal_())
            """
            sample from a normal distribution with standard deviation = std and
            mean = mu by multiplying mean 0, std 1 sample with desired std and mu.
            We need to use some technique to sample from a normal distribution with
            known mean and variance.

            The outcome is to produce a set of N (where N is the batch size) random
            self.zdims float vectors sampled from the normal distribution with learned mu
            and std for the current input.
            """
            return eps.mul(std).add_(mu)
        else:
            """
            During inference we just produce the mean of the learned distribution
            for the current input.
            """
            return mu

    def decode(self, Z):
        h3 = self.relu(self.fc3(Z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, X):
        # Encode the input vector and produce our two distribution vectors
        mu, logvar = self.encode(X.view(-1, 784))
        z = self.reparam(mu, logvar)

        return (self.decode(z), mu, logvar)

"""An RNN written as an iterated block map.

The whole module is one equation.  Writing $z$ for the state, split into slots,

    z_{t+1} = (A . b . M)^K . Inject_{t+1} . z_t                            (*)

where

    M          a Sequential2D block map, the "matrix" of (possibly nonlinear) blocks
    b          a bias, one vector per slot
    A          an activation, one module per slot
    Inject     overwrite the input slot with the next element of the sequence
    K          how many times the internal map runs per input token

Reading (*) right to left: one token arrives and is written into the input slot;
then the internal map A.b.M is applied K times; then the next token arrives.

Two timescales.  `Inject` ticks once per token (slow, external).  The internal
map ticks K times per token (fast, internal).  K = 1 collapses (*) to an
ordinary RNN, which is why `from_rnn` below can reproduce `torch.nn.RNN`
exactly.  K > 1 is the interesting case and is the reason this module exists.

See OVERVIEW_RNN_SEQUENTIAL_2D.md for the derivation, the correspondence with
torch.nn.RNN, and the reasoning behind each design decision.  Section numbers
in the comments below refer to that document.

Reference: Hershey, Paffenroth, Pathak & Tavener, "Rethinking the Relationship
between Recurrent and Non-Recurrent Neural Networks: A Study in Sparsity",
arXiv:2404.00880.  Equation numbers refer to that paper.
"""

import torch

from iterativennsimple.Sequential2D import Sequential2D, Identity


class Sequential2DRNN(torch.nn.Module):
    """A discrete dynamical system built from a Sequential2D block map.

    The state is a list of slots, `z = [z_0, z_1, ..., z_{n-1}]`, each of shape
    `(batch, features_list[i])`.  Because the map is *iterated*, it must send the
    state space to itself, so the block map's input and output partitions agree
    -- a single `features_list` describes both.

    A slot may be `None`, meaning "not yet alive": nothing has written to it.
    `None` propagates through `b` and `A` untouched (Sec. 8.4).  This is a
    deliberate choice and is *not* the same as the slot holding a zero vector,
    since `A(0 + b)` is generally neither zero nor `None`.

    Args:
        features_list: sizes of the state slots, e.g. `[input_size, hidden_size]`.
        blocks: 2D list of modules, `blocks[i][j]` maps slot i to slot j, or
            `None` for an absent block.  NOTE the index order: `i` is the
            *input* slot and `j` is the *output* slot, which is the transpose of
            the block matrix as written in the paper (Sec. 3.2).
        bias: list of bias vectors, one per slot, entries may be `None`.
        activation: list of modules, one per slot, applied after the bias.
        inject_slot: which slot `Inject` overwrites with the input.
        hidden_slot: which slot carries `h_0` / `h_n`, for the torch.nn.RNN API.
        output_slot: which slot is stacked into `output`.  Mind the one-step lag
            if this is a readout slot rather than the hidden slot (Sec. 8.3).
        K: internal iterations per input token.
        batch_first: as in torch.nn.RNN -- if True, `input` is
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
        check_input_persistence: assert that the input slot is an exact identity
            wire across the K internal iterations (Sec. 8.6).  Set False to
            deliberately experiment with a learnable/biased/squashed input slot.
    """

    def __init__(self, features_list, blocks, bias, activation,
                 inject_slot=0, hidden_slot=-1, output_slot=-1, K=1,
                 batch_first=False, check_input_persistence=True):
        super().__init__()

        n = len(features_list)
        self.features_list = features_list
        self.inject_slot = inject_slot % n
        self.hidden_slot = hidden_slot % n
        self.output_slot = output_slot % n
        self.K = K
        self.batch_first = batch_first

        # The map is iterated, so it must send the state space to itself.
        self.M = Sequential2D(features_list, features_list, blocks)

        # b: one bias per slot.  Slots without a bias are simply absent from the
        # dict, mirroring how Sequential2D stores its blocks.
        self.b = torch.nn.ParameterDict()
        for i, b_i in enumerate(bias):
            if b_i is not None:
                self.b[str(i)] = torch.nn.Parameter(b_i)

        # A: one module per slot.  Use torch.nn.Identity() for "no activation".
        self.A = torch.nn.ModuleList(activation)

        if check_input_persistence:
            self._check_input_persistence()

    def _check_input_persistence(self):
        """The input slot must be an exact identity wire (Sec. 8.6).

        With M_xx = I, b_x = 0 and A_x = id, the injected token is held
        unchanged for all K internal iterations.  Break any one of the three and
        the input drifts: a bias gives x + k*b_x, linear in K; a tanh saturates
        it.  At K = 1 the damage is invisible, because Inject overwrites the slot
        before it is ever read again -- which is exactly why this is checked at
        construction rather than left to be discovered at K = 10.
        """
        i = self.inject_slot
        assert str((i, i)) in self.M.blocks, \
            f'input slot {i} must have an Identity self-block, but it is absent'
        block = self.M.blocks[str((i, i))]
        assert isinstance(block, Identity), \
            f'input slot {i} must have an Identity self-block, got {block}'
        assert str(i) not in self.b, \
            f'input slot {i} must have no bias, or the input drifts as x + k*b'
        assert isinstance(self.A[i], torch.nn.Identity), \
            f'input slot {i} must have no activation, got {self.A[i]}'

    def internal_step(self, z):
        """One application of the internal map, `A . b . M`.

        This is the fast timescale.  Order matters: the activation acts on the
        *sum* of everything arriving at a slot, which is what produces
        `tanh(W_xh x + W_hh h + b)` rather than `tanh(W_xh x) + tanh(W_hh h)`
        (Sec. 5.4).
        """
        z = self.M.forward_list(z)                                    # M
        z = [z_i if z_i is None or str(i) not in self.b               # b
             else z_i + self.b[str(i)]
             for i, z_i in enumerate(z)]
        z = [z_i if z_i is None else A_i(z_i)                         # A
             for z_i, A_i in zip(z, self.A)]
        return z

    def external_step(self, z, x_t):
        """One token: inject, then run the internal map K times.

        This is a single application of (*), and is the slow timescale.
        """
        z = list(z)
        z[self.inject_slot] = x_t                                     # Inject
        for _ in range(self.K):
            z = self.internal_step(z)
        return z

    def forward(self, input, h_0=None):
        """Run the system over a sequence, with the torch.nn.RNN signature.

        Args:
            input: `(seq, batch, input_size)`, or `(batch, seq, input_size)` if
                `batch_first`.
            h_0: `(1, batch, hidden_size)`, or `None` for zeros.  The leading 1
                is torch.nn.RNN's `num_layers * num_directions`, which is always
                1 here (Sec. 8.5).

        Returns:
            output: the output slot at every step, same layout as `input`.
            h_n: `(1, batch, hidden_size)`, the final hidden slot.
        """
        if self.batch_first:
            input = input.transpose(0, 1)          # -> (seq, batch, input_size)
        seq_len, batch_size = input.shape[0], input.shape[1]

        z = [None] * len(self.features_list)
        if h_0 is None:
            z[self.hidden_slot] = torch.zeros(
                batch_size, self.features_list[self.hidden_slot],
                dtype=input.dtype, device=input.device)
        else:
            z[self.hidden_slot] = h_0[0]           # (1, batch, h) -> (batch, h)

        outputs = []
        for t in range(seq_len):
            z = self.external_step(z, input[t])
            outputs.append(z[self.output_slot])

        output = torch.stack(outputs, dim=0)       # (seq, batch, out)
        h_n = z[self.hidden_slot].unsqueeze(0)     # (1, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_rnn(cls, rnn, K=1):
        """Copy the weights of a torch.nn.RNN into the two-slot map of eq. (9).

        With state `z = [x, h]` the block map is, in the paper's row=output
        convention,

            M = [[ I    , None ],          x <- x
                 [ W_xh , W_hh ]]          h <- W_xh x + W_hh h

        together with `b = [None, b_ih + b_hh]` and `A = [id, tanh]`.  At K = 1
        this reproduces the torch.nn.RNN recurrence exactly,

            h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh).

        No transposes anywhere: `torch.nn.Linear.weight` and `rnn.weight_ih_l0`
        are both stored as `(out_features, in_features)`, so the weights copy
        straight across (Sec. 3.3).

        PyTorch carries two bias vectors where the map has one per slot.  Their
        sum is the only thing the forward pass can see, so outputs agree exactly
        -- but the map back to a torch.nn.RNN state_dict is not unique
        (Sec. 8.2).

        Passing K > 1 keeps the weights and runs the internal map K times per
        token.  That is no longer a torch.nn.RNN, and is the point.
        """
        assert rnn.num_layers == 1, 'multi-layer stacking is out of scope (Sec. 8.5)'
        assert not rnn.bidirectional, 'bidirectional is not implemented yet'

        input_size, hidden_size = rnn.input_size, rnn.hidden_size
        sigma = {'tanh': torch.nn.Tanh, 'relu': torch.nn.ReLU}[rnn.nonlinearity]()

        # Inherit dtype and device from the source, so from_rnn(rnn.double())
        # gives a float64 map rather than silently casting the weights down.
        like = dict(dtype=rnn.weight_ih_l0.dtype, device=rnn.weight_ih_l0.device)

        W_xh = torch.nn.Linear(input_size, hidden_size, bias=False, **like)
        W_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False, **like)
        with torch.no_grad():
            W_xh.weight.copy_(rnn.weight_ih_l0)
            W_hh.weight.copy_(rnn.weight_hh_l0)
            b_h = (rnn.bias_ih_l0 + rnn.bias_hh_l0).clone()

        # blocks[i][j] maps slot i to slot j; slots are x = 0, h = 1.
        blocks = [[Identity(in_features=input_size, out_features=input_size), W_xh],
                  [None,                                                      W_hh]]

        return cls(features_list=[input_size, hidden_size],
                   blocks=blocks,
                   bias=[None, b_h],
                   activation=[torch.nn.Identity(), sigma],
                   inject_slot=0, hidden_slot=1, output_slot=1,
                   K=K, batch_first=rnn.batch_first)

    @classmethod
    def from_3x3(cls, input_size, output_size, hidden_size,
                 W_xy=None, W_yy=None, W_hy=None,
                 W_xh=None, W_yh=None, W_hh=None,
                 A_y=None, A_h=None, bias=True, K=1, batch_first=False):
        """The general three-slot map, state `z = [x, y, h]`.

        Six free blocks -- the y-row and the h-row of the 3x3 matrix:

            M = [[ I    , None , None ],   x <- x                (fixed, Sec. 8.6)
                 [ W_xy , W_yy , W_hy ],   y <- W_xy x + W_yy y + W_hy h
                 [ W_xh , W_yh , W_hh ]]   h <- W_xh x + W_yh y + W_hh h

        Naming is source-first: `W_ab` maps slot a to slot b (Sec. 3.4).

        The x-row is not exposed.  `M_xx = I` is fixed to keep the injected token
        available at every internal iteration -- an input skip connection across
        the K unrolled steps, and what makes the K -> infinity limit well posed
        (Sec. 8.6).  The other two x-row cells are deferred (Sec. 10.1).

        Blocks are arbitrary modules carrying `in_features` / `out_features`, so
        `MaskedLinear`, `MonarchLinear`, `SparseLinear`, `Sequential1D` and even
        a nested `Sequential2D` all drop straight in.  `None` means absent.

        Two things to watch:

        * Blocks must be bias-free.  Bias belongs to the slot, not the block
          (Sec. 8.2), so a block carrying its own bias double-counts.
        * A nested block does not receive the outer activation.  `A` acts on the
          slot sum, after every block has contributed; anything inside a block
          must carry its own nonlinearity.

        Note `W_hy` is read one internal iteration behind `W_hh` -- the update is
        Jacobi, so every block reads the *old* state (Sec. 8.3).  Reading the
        y-slot therefore gives a one-step-delayed readout of h.

        This will generally not agree with a torch.nn.RNN.  That is the point.
        """
        for name, W in [('W_xy', W_xy), ('W_yy', W_yy), ('W_hy', W_hy),
                        ('W_xh', W_xh), ('W_yh', W_yh), ('W_hh', W_hh)]:
            assert getattr(W, 'bias', None) is None, \
                f'{name} must be bias-free; bias belongs to the slot (Sec. 8.2)'

        # blocks[i][j] maps slot i to slot j; slots are x = 0, y = 1, h = 2.
        blocks = [[Identity(in_features=input_size, out_features=input_size), W_xy, W_xh],
                  [None,                                                      W_yy, W_yh],
                  [None,                                                      W_hy, W_hh]]

        b = [None,
             torch.zeros(output_size) if bias else None,
             torch.zeros(hidden_size) if bias else None]

        return cls(features_list=[input_size, output_size, hidden_size],
                   blocks=blocks,
                   bias=b,
                   activation=[torch.nn.Identity(),
                               A_y if A_y is not None else torch.nn.Identity(),
                               A_h if A_h is not None else torch.nn.Tanh()],
                   inject_slot=0, hidden_slot=2, output_slot=1,
                   K=K, batch_first=batch_first)

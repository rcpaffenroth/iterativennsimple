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

See tasks/OVERVIEW_RNN_SEQUENTIAL_2D.md for the derivation, the correspondence with
torch.nn.RNN, and the reasoning behind each design decision.  Section numbers
in the comments below refer to that document.

Reference: Hershey, Paffenroth, Pathak & Tavener, "Rethinking the Relationship
between Recurrent and Non-Recurrent Neural Networks: A Study in Sparsity",
arXiv:2404.00880.  Equation numbers refer to that paper.
"""

import torch

from torch.nn.utils.rnn import PackedSequence

from iterativennsimple.Sequential2D import Sequential2D, Identity


def _check_block(name, block, in_features, out_features):
    """Sanity-check one block of the map, before Sequential2D ever sees it.

    Catches, in order of how likely each is to actually happen:

    1.  A transposed block -- `Linear(hidden, input)` where `Linear(input,
        hidden)` was meant.  `Sequential2D` would catch this too, but its message
        talks about slot indices rather than about which W you got backwards.
    2.  A weight matrix whose actual shape disagrees with the `in_features` /
        `out_features` the module advertises.  PyTorch stores weights as
        `(out_features, in_features)`, so a genuinely transposed weight shows up
        here even when the declared sizes look right.
    3.  A block carrying its own bias, which would double-count against the slot
        bias (Sec. 8.2).

    Not all block types have a `.weight` -- MonarchLinear keeps factors, a nested
    Sequential2D keeps sub-blocks -- so check 2 only applies when there is one.
    """
    if block is None:
        return

    assert getattr(block, 'in_features', None) == in_features \
        and getattr(block, 'out_features', None) == out_features, (
        f'{name} should map {in_features} -> {out_features} features, but it '
        f'declares {getattr(block, "in_features", None)} -> '
        f'{getattr(block, "out_features", None)}.  Naming is source-first, so '
        f'{name} takes slot "{name[2]}" and produces slot "{name[3]}" -- if those '
        f'look swapped, that is the bug.')

    weight = getattr(block, 'weight', None)
    if isinstance(weight, torch.Tensor):
        assert weight.ndim == 2, \
            f'{name}.weight should be a matrix, but it has {weight.ndim} dimensions'
        assert tuple(weight.shape) == (out_features, in_features), (
            f'{name}.weight has shape {tuple(weight.shape)} but should be '
            f'{(out_features, in_features)}.  PyTorch stores weights as '
            f'(out_features, in_features) -- looks like a transpose.')

    assert getattr(block, 'bias', None) is None, \
        f'{name} must be bias-free; bias belongs to the slot, not the block (Sec. 8.2)'


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
        observation: optional module `g`, applied to the output slot on the way
            out.  This is the *observation map* of the classical state-space
            form, and is not part of the dynamics -- see `internal_step`.
        K: internal iterations per input token.
        batch_first: as in torch.nn.RNN -- if True, `input` is
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
        check_input_persistence: assert that the input slot is an exact identity
            wire across the K internal iterations (Sec. 8.6).  Set False to
            deliberately experiment with a learnable/biased/squashed input slot.
    """

    def __init__(self, features_list, blocks, bias, activation,
                 inject_slot=0, hidden_slot=-1, output_slot=-1, observation=None,
                 K=1, batch_first=False, check_input_persistence=True):
        super().__init__()

        n = len(features_list)
        self.features_list = features_list
        self.inject_slot = inject_slot % n
        self.hidden_slot = hidden_slot % n
        self.output_slot = output_slot % n
        self.observation = observation
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

    def observe(self, z):
        """The observation map `g`, reading the output slot: y_t = g(z_t).

        This is the second half of the classical state-space form

            z_{t+1} = F(z_t, x_{t+1})       the state map, (*) above
            y_t     = g(z_t)                the observation map, here

        which is how S4, MAMBA and ordinary linear control are written -- their
        `C` matrix is an observation, not a state component.  Keeping it separate
        matters for two reasons:

        *   **No lag.**  `g` reads the state as it is *now*, so a readout via `g`
            gives y_t = g(h_t).  A readout carried in a *slot* instead goes through
            the block map, which is a Jacobi update, so it lands one internal
            iteration behind (Sec. 8.3).  At K = 1 that is a full token of delay.
        *   **No feedback.**  `g` cannot influence the dynamics, by construction.
            When feedback *is* wanted -- the paper's above-diagonal block S,
            Sec. 6.4 -- the readout has to be a slot, and then the lag is the price.

        Because `g` sits outside the block matrix, the "bias belongs to the slot"
        rule (Sec. 8.2) does not apply to it: an ordinary `torch.nn.Linear` with
        its own bias is exactly right here.
        """
        y = z[self.output_slot]
        assert y is not None, (
            f'slot {self.output_slot} is the output slot but nothing ever writes to '
            f'it, so it is still None (Sec. 8.4).  Give it an incoming block, or '
            f'point output_slot somewhere else.')
        return y if self.observation is None else self.observation(y)

    def initial_state(self, batch_size, h_0, dtype, device):
        """The state before the first token: every slot None except the hidden one.

        `None` means "not yet alive" (Sec. 8.4), so slots nothing has written to
        stay out of the arithmetic entirely rather than contributing zeros.
        """
        z = [None] * len(self.features_list)
        if h_0 is None:
            z[self.hidden_slot] = torch.zeros(
                batch_size, self.features_list[self.hidden_slot],
                dtype=dtype, device=device)
        else:
            z[self.hidden_slot] = h_0[0]           # (1, batch, h) -> (batch, h)
        return z

    def forward(self, input, h_0=None):
        """Run the system over a sequence, with the torch.nn.RNN signature.

        Args:
            input: `(seq, batch, input_size)`, or `(batch, seq, input_size)` if
                `batch_first`, or a `PackedSequence`.
            h_0: `(1, batch, hidden_size)`, or `None` for zeros.  The leading 1
                is torch.nn.RNN's `num_layers * num_directions`, which is always
                1 here (Sec. 8.5).

        Returns:
            output: the observation at every step, same layout as `input`.
            h_n: `(1, batch, hidden_size)`, the final hidden slot.
        """
        if isinstance(input, PackedSequence):
            return self.forward_packed(input, h_0)

        if self.batch_first:
            input = input.transpose(0, 1)          # -> (seq, batch, input_size)
        seq_len, batch_size = input.shape[0], input.shape[1]

        z = self.initial_state(batch_size, h_0, input.dtype, input.device)

        outputs = []
        for t in range(seq_len):
            z = self.external_step(z, input[t])       # z_{t+1} = F(z_t, x_{t+1})
            outputs.append(self.observe(z))           # y_t     = g(z_t)

        output = torch.stack(outputs, dim=0)       # (seq, batch, out)
        h_n = z[self.hidden_slot].unsqueeze(0)     # (1, batch, hidden)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_n

    def forward_packed(self, input, h_0=None):
        """Run the system over a `PackedSequence` of variable-length sequences.

        Nothing about the dynamics changes here.  `M`, `b`, `A`, `Inject`,
        `internal_step` and `external_step` are untouched and know nothing about
        sequences -- they act on one already-assembled batch of slot tensors.  All
        that varies is *how many rows of the batch are still alive* at each step,
        so the whole of packing is confined to this method.

        How a PackedSequence works, since the layout is doing the real work here:
        sequences are sorted by decreasing length, `batch_sizes[t]` counts how many
        are still running at step t (so it is non-increasing), and `data` is the
        timesteps laid end to end -- `data[offset : offset + batch_sizes[t]]` is
        step t for the sequences still alive.  Because the sort is by decreasing
        length, the ones that finish at step t are always a *suffix* of the batch,
        which is what makes truncating to a prefix the right move.

        Args:
            input: a `PackedSequence`.  `batch_first` is irrelevant -- packed data
                is already flattened -- and is ignored, as torch.nn.RNN ignores it.
            h_0: `(1, batch, hidden_size)` in the caller's original ordering, or
                `None` for zeros.

        Returns:
            output: a `PackedSequence` with the same `batch_sizes`.
            h_n: `(1, batch, hidden_size)`, each sequence's hidden state at *its
                own* final step, back in the caller's original ordering.
        """
        data, batch_sizes = input.data, input.batch_sizes
        max_batch = int(batch_sizes[0])            # all sequences alive at step 0

        # `data` is in sorted order but the caller's h_0 is not, so line them up.
        if h_0 is not None and input.sorted_indices is not None:
            h_0 = h_0.index_select(1, input.sorted_indices)

        z = self.initial_state(max_batch, h_0, data.dtype, data.device)

        # Each sequence's final hidden state, filled in as sequences drop out.
        h_final = torch.zeros(max_batch, self.features_list[self.hidden_slot],
                              dtype=data.dtype, device=data.device)

        outputs = []
        alive = max_batch
        offset = 0
        for batch_size in batch_sizes.tolist():
            if batch_size < alive:
                # Sequences [batch_size, alive) ended at the previous step, so the
                # state they hold right now is their final one.  Harvest it before
                # truncating, or it is gone.
                h_final[batch_size:alive] = z[self.hidden_slot][batch_size:alive]
                z = [None if slot is None else slot[:batch_size] for slot in z]
                alive = batch_size

            x_t = data[offset:offset + batch_size]
            offset += batch_size

            z = self.external_step(z, x_t)         # z_{t+1} = F(z_t, x_{t+1})
            outputs.append(self.observe(z))        # y_t     = g(z_t)

        # Whatever is still running reached the end of the longest sequence.
        h_final[:alive] = z[self.hidden_slot][:alive]

        # Concatenating along the batch axis rebuilds exactly the packed layout.
        output = PackedSequence(torch.cat(outputs, dim=0), batch_sizes,
                                input.sorted_indices, input.unsorted_indices)

        if input.unsorted_indices is not None:
            h_final = h_final.index_select(0, input.unsorted_indices)
        return output, h_final.unsqueeze(0)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_rnn(cls, rnn, K=1, readout=None):
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

        There is no y-slot, because torch.nn.RNN has no readout: its `output` *is*
        the sequence of hidden states, and there are no weights in the source RNN
        to copy into one.  If you want the readout you would normally compose
        afterwards -- `torch.nn.Linear(hidden_size, n_classes)` -- pass it as
        `readout` and it becomes the observation map `g` (see `observe`), giving
        y_t = readout(h_t) with no lag.  With `readout=None` the output is h_t
        itself, which is what makes this bit-comparable with torch.nn.RNN.
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
                   observation=readout,
                   K=K, batch_first=rnn.batch_first)

    @classmethod
    def from_3x3(cls, input_size, output_size, hidden_size, *,
                 W_xy=None, W_yy=None, W_hy=None,
                 W_xh=None, W_yh=None, W_hh=None,
                 A_y=None, A_h=None, bias=True, readout=None,
                 K=1, batch_first=False):
        """The general three-slot map, state `z = [x, y, h]`.

        Six free blocks -- the y-row and the h-row of the 3x3 matrix:

            M = [[ I    , None , None ],   x <- x                (fixed, Sec. 8.6)
                 [ W_xy , W_yy , W_hy ],   y <- W_xy x + W_yy y + W_hy h
                 [ W_xh , W_yh , W_hh ]]   h <- W_xh x + W_yh y + W_hh h

        Naming is source-first: `W_ab` maps slot a to slot b (Sec. 3.4).  The six
        blocks are **keyword-only**, deliberately: the source-first/target-first
        distinction has been got wrong once already, and passing them positionally
        would let a reversed name land a block silently in the wrong cell.  As
        keywords, a wrong name is a TypeError.

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

        There are two different ways to get a y out, and they are not
        interchangeable:

        *   **As a state slot**, via `W_hy`.  The y-slot then takes part in the
            dynamics and can feed back into h through `W_yh` -- the paper's
            above-diagonal block S, Sec. 6.4.  The cost is a lag: the update is
            Jacobi, so every block reads the *old* state, and the y-slot is one
            internal iteration behind h (Sec. 8.3).  At K = 1 that is a full token
            of delay; as K grows and the iteration converges it shrinks, since
            h^(K-1) -> h^(K).
        *   **As an observation**, via `readout`.  This is `g` in the state-space
            form (see `observe`), reads h as it is now, and has no lag -- but
            cannot feed back, by construction.  Passing it moves `output` off the
            y-slot and onto `readout(h_t)`.

        Use `readout` unless you specifically want feedback.  Both can coexist:
        the y-slot may still carry `W_yh` feedback while `readout` supplies the
        reported output.

        This will generally not agree with a torch.nn.RNN.  That is the point.
        """
        # (source size, target size) for each block, in source-first naming.
        expected = {
            'W_xy': (input_size,  output_size), 'W_xh': (input_size,  hidden_size),
            'W_yy': (output_size, output_size), 'W_yh': (output_size, hidden_size),
            'W_hy': (hidden_size, output_size), 'W_hh': (hidden_size, hidden_size),
        }
        for name, W in [('W_xy', W_xy), ('W_yy', W_yy), ('W_hy', W_hy),
                        ('W_xh', W_xh), ('W_yh', W_yh), ('W_hh', W_hh)]:
            _check_block(name, W, *expected[name])

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
                   inject_slot=0, hidden_slot=2,
                   # An observation reads h directly; a slot readout reads y.
                   output_slot=2 if readout is not None else 1,
                   observation=readout,
                   K=K, batch_first=batch_first)

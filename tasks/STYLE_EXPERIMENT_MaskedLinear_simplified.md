# Style experiment: four reimplementations of `MaskedLinear`

**Purpose.** RCP finds some assistant-written code opaque and wants to pin down
*why*. Reimplementations of the same module were written to be judged against each
other; the judgement then becomes a rule in `PRINCIPLES.md` Part 1 so that it does
not have to be re-argued every session.

**Status: settled.** A and B were written first, as the two poles. C came next, at
RCP's request, as the cross of the two axes I had predicted mattered — a test of my
model of RCP's taste rather than an independent entrant. D changed the rules: RCP
licensed it to **sacrifice functionality for readability**, so it was never a
drop-in replacement and was never commensurable with the others.

**RCP kept D and deleted A, B and C.** D now lives at
`iterativennsimple/MaskedLinear_simplified.py`. §7 records what that settles and
what it does not; the *reason* for the choice is still RCP's to write, and this
document's sections 1-6 are kept as the record of what was measured.

| | |
| --- | --- |
| `iterativennsimple/MaskedLinear.py` | the original, untouched; still the reference implementation |
| `iterativennsimple/MaskedLinear_simplified.py` | **D**, kept — the same mathematics with the string interpreter removed; **not a drop-in** |
| `tests/test_MaskedLinear_simplified.py` | agreement with the reference, plus worked usage |
| ~~`MaskedLinear_A.py`~~, ~~`_B.py`~~, ~~`_C.py`~~ | **A, B, C — deleted.** Described in §2, measured in §2-§3; the numbers stand, the files are gone |

---

## 1. What is held fixed, so the comparison is about style and nothing else

Per `PRINCIPLES.md` §2.10, the axes are enumerated before anything is compared.
These are identical in A, B and C, so they cannot explain a preference between
them. **D holds items 2 through 7 and deliberately breaks item 1**; its cuts are
listed in §2.

1. The public API: class name, constructor signature, the attributes `weight_0`,
   `U`, `mask`, `bias`, and the methods `from_description`, `from_config`,
   `from_coo`, `from_MLP`, `from_optimal_linear`, `reset_parameters`, `forward`,
   `extra_repr`, `number_of_trainable_parameters`.
2. `weight_0` and `mask` stay `Parameter(requires_grad=False)` rather than becoming
   `register_buffer`. Buffers are the idiomatic way to say "fixed tensor that is
   part of the state", and they would be my choice in new code, but
   `tests/test_Sequential2D.py:88` asserts `len(list(model.parameters())) == 20`
   and that number counts `weight_0` and `mask`. Changing it is a separate
   decision, not a style one.
3. `trainable='non-zero'` keys off the *realised values* (`values != 0`), not the
   sparsity pattern. These differ only for an initialiser that produces an exact
   zero inside the pattern, and `SparseLinear.from_MaskedLinearExact` assumes the
   value-based rule (see its comment at `SparseLinear.py:310`).
4. `from_optimal_linear` still solves the normal equations rather than using
   `torch.linalg.lstsq`, so the numbers reproduce exactly.
5. The random block types draw from `torch`, not `numpy`, so `torch.manual_seed`
   now reproduces them — it did not before. The consequence is that random block
   types do **not** reproduce the original's numbers; nothing checked in depends on
   them, but old notebook output will move.
6. The per-entry `np.fromfunction(np.vectorize(...))` construction is gone in all
   three, replaced by whole-block tensor operations. Building one 512x512 `'R=0.5'`
   block with `'G'` initialisation: **0.143 s original, 0.006 s (A), 0.006 s (B),
   0.004 s (C)** [det, one run each], a factor of ~24.
7. The `time.perf_counter()` / `logger.debug` pair inside `forward` is gone in
   all three. It cost two clock reads and an f-string per forward pass. At
   512x512, batch 128, forward is 297 us original vs 281 us (A) and 277 us (B)
   [det, best of 20] — i.e. this is a legibility change, not a speed one; do not read the 6% as
   the logging's cost at that size.

All four files also state items 2-6 in their own module docstring, per `PRINCIPLES.md`
§2.9: a decision recorded only here is a decision the next reader of the code will
not see.

## 2. What differs — the actual experiment

Two axes, crossed three ways out of four. **Where the fixed matrices get written**
(slice assignment into an allocated layer, vs. assemble a `(W_0, Omega)` pair and
construct once) and **how the two string mini-languages get dispatched** (a flat
`if/elif` chain read where it is used, vs. parsed into named types and dispatched
with `match`).

| | flat `if/elif` dispatch | parse-to-types + `match` |
| --- | --- | --- |
| **write into slices** | **A** | not written |
| **assemble a pair, construct once** | **C** | **B** |

**A — "two matrices, written into by slices."** One organising idea: the layer is
determined by the pair of dense matrices `(W_0, Omega)`, so every constructor
allocates the layer and assigns into slices of `A.weight_0` and `A.mask`. Two
module-level helpers, `_support` and `_values`, each a flat `if/elif` chain over
the string mini-language, each one screen. The block offsets are computed with
`accumulate` and appear as explicit `slice(...)` objects next to the mathematics
that fills them.

**B — "a `(values, mask)` pair, and a vocabulary of specifications."** Two
organising ideas. First, `Block` is a `NamedTuple` of two equal-shaped matrices,
and a 2D array of `Block`s assembles into one `Block` because assembly acts on
the two matrices independently; every constructor assembles `Block`s and hands the
result to `from_block`, the single place that writes into the module. Second, the
string mini-language is parsed *once* into ten small frozen dataclasses (`Full`,
`Diagonal`, `Bernoulli(p)`, `PerRow(n)`, `Gaussian(mu, sigma)`, ...) and everything
downstream dispatches with `match` on those types rather than on string prefixes.

**C — B's first idea without its second.** `Block`, `block_matrix`, `frozen` /
`trainable` / `trainable_where_nonzero`, and `from_block` as the one writer, exactly
as in B; `support` and `values` as two flat `if/elif` functions, exactly as in A.
Nothing else differs from either.

Measured shape of the four (docstrings excluded from "code lines"):

| | original | A | B | C | D |
| --- | --- | --- | --- | --- | --- |
| total lines | 451 | 337 | 437 | 376 | 287 |
| code lines | 231 | 164 | 219 | 166 | **70** |
| docstring lines | 159 | 134 | 147 | 156 | 170 |
| classes | 1 | 1 | 12 | 2 | 2 |
| functions | 19 | 12 | 21 | 19 | 15 |
| longest function | 102 | 56 | 28 | 36 | 25 |
| max nesting depth | 7 | 4 | 2 | 2 | **1** |

That table is the trade in one place. **B buys small functions and shallow nesting
by adding names to keep track of; A buys few names by making each function longer.**
A's `from_description` is 56 lines you read once, top to bottom. B's is 12, but
answering "what does `'Row=3'` do" means `parse_pattern` then `support`, two hops.

D's column is the one to look at twice: **70 code lines against the original's
231, and more docstring than code.** Seven-tenths of that file was the string
interpreter and the plumbing around it. Nothing in D nests more than one level
deep. The price is in §3: 20 of the 58 existing tests stop even running.

C says the two halves of that trade are separable, and the measurement is the
interesting part: **C is A's code volume (166 lines vs 164) with B's flatness
(nesting 2, longest function 36 lines, and that 36 is 13 lines of docstring
listing the mini-language).** So `Block`/`block_matrix` is close to free — it pays for
itself by deleting the offset arithmetic it replaces — whereas the ten spec types
cost B 53 code lines and ten names, and buy dispatch on types instead of string
prefixes.

**D is off this grid.** It keeps C's `Block`/`block_matrix` and drops the two
mini-languages outright, together with everything that existed to serve them. Its
`from_description` replacement is to write the block matrix down:

    MaskedLinear(block_matrix([
        [frozen(torch.zeros(5, 6)),                 trainable(torch.randn(5, 8) * 0.7 + 0.2)],
        [trainable_where_nonzero(0.3 * torch.eye(7, 6)),
         trainable_where_nonzero(torch.rand(7, 8).mul(2).sub(1) * bernoulli(7, 8, p=0.5))],
    ]))

Three parallel 2D arrays that had to stay aligned become one; the shapes are
visible instead of implied by position; any cell may be any tensor; values are
written with `torch` directly (`torch.randn(5, 8) * 0.7 + 0.2` *is* `'G=0.2,0.7'`),
so only the three sparsity patterns torch does not provide survive as functions.
The full cut list with a restore cost for each is in D's own module docstring; the
summary is: `from_description`, `from_config` (called from nowhere in the
repository [det, grep]), `from_coo` (a one-line body), `reset_parameters`
(inlined), the `device`/`dtype` pass-through arguments, and
`MaskedLinear(in_features, out_features)` — D's constructor takes the pair it is
defined by, and `MaskedLinear.dense(in, out)` is the old behaviour.

What C gives up is the same thing B gives up, and A does not: **seed
compatibility.** Drawing the blocks before constructing the layer means
`__init__`'s own kaiming draw for `weight_0` — immediately overwritten, but still
consumed — shifts the random stream, so `from_MLP` under a recorded seed gives
different (equally distributed) weights. B and C are identical to each other here
and both differ from the original [det, `test_only_A_is_seed_compatible_with_the_original`].

## 3. Evidence that the mathematics is unchanged

- At the time of the comparison, `tests/test_MaskedLinear_AB.py` ran 56 tests over
  all four implementations, all passing [det]. With A, B and C deleted this file is
  now `tests/test_MaskedLinear_simplified.py` and runs 16: the agreement checks
  below, and a section of worked usage. Shared behaviour was
  parametrised over all four implementations; exact entrywise agreement with the
  original is asserted for every deterministic construction (`__init__`,
  `from_MLP`'s structure, `from_coo`, `from_optimal_linear`, a fully deterministic
  `from_description`, and the single-block integer path `Sequential2D` uses); for
  the random block types, only the structure of the sparsity pattern is compared,
  because the generators and stream positions differ.
- Drop-in substitution: with A, with B, or with C substituted for the original,
  `test_MaskedLinear.py`, `test_Sequential2D.py` and `test_SparseLinear.py` give
  **58 passed**, the same as the original [det].
- **D substituted for the original: 38 passed, 20 failed** [det]. That number *is*
  the sacrifice, measured rather than asserted, and it is the reason `MaskedLinear.py`
  stays where it is rather than being replaced. Every one of the 20 fails for one
  of three reasons — a call to `MaskedLinear(in_features, out_features)`, a call to
  `from_description`, or a call to `from_coo` — and none of them for a
  mathematical disagreement. `Sequential2D` is among the casualties.
- D's mathematics is checked separately, against the reference, in
  `tests/test_MaskedLinear_simplified.py` [det]: `dense` reproduces `torch.nn.Linear` entrywise under one
  seed; the gradient still lands on `U`; `from_MLP` and `from_optimal_linear` agree
  with the original; the one-line replacement for `from_coo` applies the COO
  matrix; and `test_D_reproduces_a_description_it_can_no_longer_parse` writes out
  the block matrix of a deterministic `from_description` call by hand and gets the
  same `W_0` and `Omega` entry for entry. Total across all four: **56 tests
  passing.**

## 4. Findings about the original, found on the way

1. **Dispatch depends on check order.** `_getBlock` tests `block_type[0:3] == "Row"`
   before `block_type[0] == "R"`; reorder those two branches and `'Row=3'` silently
   becomes a Bernoulli block with `p = float('w=3')`, i.e. a crash, but only by
   luck. All three reimplementations match on the full prefix (`"R="` vs `"Row="`), which
   makes the branches order-independent.
2. **`'D'` blocks only work through `from_description`.** `_getBlock` calls
   `torch.min(out_features, in_features)` on what must therefore be tensors; it is
   tensors only because `from_description` converts the size lists with
   `torch.tensor`. A `'D'` block reached with plain Python ints raises [det]. All
   three reimplementations use `torch.eye(out_features, in_features)`, which does
   not care.
3. **`from_coo(check_mask=True)` cannot ever have run**, for two reasons:
   `for idx in range(coo.indices())` calls `range` on a tensor (`TypeError`), and
   `.indices()` itself raises on an uncoalesced tensor, which is what
   `torch.sparse_coo_tensor` returns. All three reimplementations iterate
   `coo.coalesce().indices().T` and do catch the condition the flag is for — a
   stored entry whose value is exactly 0.0, which `mask = (weight_0 != 0)` misses
   and which is therefore silently frozen [det, verified both ways].
4. **The module docstring emits `SyntaxWarning`** — `:math:\`A = U \cdot \Omega\``
   in a non-raw string makes `\c` an invalid escape. Two warnings at import.
5. **`from_optimal_linear` leaves `Omega = 0`**, so a layer built that way has no
   trainable parameters at all except the bias. That is presumably intended (it is
   an initialisation, and `test_fromOptimalLinear` asserts the output is *not* close
   to the truth), but it is nowhere stated. All three reimplementations say so.
6. **`test_fromOptimalLinear` asserts a negative** (`assert not torch.all(...)`),
   which passes for the wrong-looking reason: the map is exact in exact arithmetic
   and misses by 5.7e-6 in float32, just over `isclose`'s tolerance. Using
   `torch.linalg.lstsq` instead gives 3.8e-6 and the test still passes [det]; with
   `cond(X) = 2.2` the normal equations are not actually hurting anything here. So
   the stable solve could be adopted whenever wanted — my earlier note that a test
   depended on the unstable numbers was wrong, and the files now say so correctly.

## 5. My own preference, recorded before the judgement

Pre-registered before RCP read any of the files, so it cannot be reverse-engineered
from his answer. Verbatim from the version written with only A and B on disk:

> I expect to prefer **A** for this file, and I expect the reason to be that B's
> parse-then-`match` layer is two hops where one would do: the mini-language is
> used in exactly one place, so naming its cases buys documentation that a
> docstring already provides. Ten dataclasses is ten things to hold.
>
> But B has the better *idea* in it, and it is not the dataclasses — it is `Block`
> and `stack`. [...] So my guess at the right answer is neither file as written:
> **B's `Block`/`stack` with A's flat dispatch.**

(`stack` is quoted as written at the time; it has since been renamed
`block_matrix`, for the reason in `PRINCIPLES.md` §1.5 — the old name was borrowed
from `torch.stack`, which adds an axis, and this does not.)

C is that guess, made concrete. The argument for it, in one comparison —
`from_optimal_linear`'s block matrix in C:

    blocks = [[frozen(torch.eye(D)),  frozen(torch.zeros(D, K))],
              [frozen(W_ls),          frozen(torch.zeros(K, K))]]

and the same thing in A:

    A.weight_0[:X_size, :X_size] = torch.eye(X_size)
    A.weight_0[X_size:, :X_size] = W_ls

The first is one line of code per line of the mathematics and says "frozen" out
loud four times; the second makes the reader compute where each block lands and
says "frozen" nowhere — `Omega` is zeroed elsewhere. Against that, A's version
allocates no intermediate matrices and reproduces old seeds.

So my ranking of A, B, C is **C, then A, then B**, and I hold it loosely: it is an
argument about reading, and RCP is the reader.

### On D, written after the ranking above

D is the file I would want to work in, and the honest reason is the 70-line
column: almost nothing in it is there to serve the code itself. Two of its cuts I
will defend hard.

- **The string mini-language belongs next to the configuration loader, not inside
  the layer.** `'R=0.5'` exists because a YAML file cannot hold a Python function.
  That is a fact about YAML, so the interpreter should sit with the YAML, in the
  one place that needs it — `Sequential2D.from_config` and the LRA harness — and
  the layer should take tensors. Every reader of `MaskedLinear` currently pays for
  a feature only the config path uses.
- **Three parallel 2D arrays that must stay aligned is a defect, not an
  interface.** `block_types`, `initialization_types` and `trainable` are indexed
  `[i][j]` in lockstep, and a misalignment is silent. `PRINCIPLES.md` §2.10 is
  about exactly this failure in configs; it applies to arguments too.

Two I am less sure of, and would not merge without RCP's say-so.

- **`MaskedLinear(in, out)` → `MaskedLinear(block)` / `MaskedLinear.dense(in, out)`.**
  It makes the constructor take the object the layer is defined by, and it removes
  the allocate-then-overwrite that costs B and C their seed compatibility. But it
  breaks `Sequential2D` and every notebook, and the gain is smaller than the
  breakage. If only one of D's cuts has to go, this is the one.
- **`device` / `dtype` dropped from every signature.** Ten parameters for a
  `.to(...)` call, but the float64 path now initialises in float32 and casts, which
  is a real if small loss of resolution.

And one criticism of D that I do not think is answerable inside the file: **it now
carries more docstring than code** — 170 lines to 70 — and a good part of that is
the cut list, which is design-record material. In a real merge the cut list belongs
in this document and D's docstring shrinks by a third. Left in place here because
the cuts *are* the experiment.

## 6. The configuration path — written, in `masked_linear_simplified_config.py`

§2 recorded that dropping `from_description` cost the YAML path, and that the fix
belonged next to the configuration files rather than inside the layer. That module
now exists: `iterativennsimple/masked_linear_simplified_config.py` imports
`MaskedLinear_simplified` and the layer knows nothing about it.

Four candidate formats were written and compared the same way the four
implementations were, by building one block matrix in each and asserting they
agree. Two were cut as dominated:

- **Strings in a grid** (`"const:0.5 / diagonal / nonzero"`), the original's idea
  cleaned up. Its parser was 22 lines and bought two lines of YAML over the dict
  form — and parsing is exactly where the original's bugs were: order-dependent
  prefix matching, floats read out of the middle of strings.
- **YAML names a Python builder** and carries only its scalar knobs. Smallest
  adapter of the four at 13 lines and the most expressive, but the configuration
  stops recording the structure, so reproducing a run needs the Python at that
  commit and a layout sweep means editing code.

The two that survived are kept, and **both**, at RCP's suggestion, which is better
than either alone:

- **the edge list** — named slots, only the non-zero blocks listed, everything
  else zero and frozen. This is the only format the loader accepts. It grows with
  the number of blocks rather than with slots², `from`/`to` remove the row/column
  orientation trap, and a structural change between two runs is a one-line diff.
- **the grid** — the same information as a block matrix, one row per output slot.
  Not loadable; `from_edges(to_edges(grid))` builds one. It exists to be *looked
  at*, and `python -m iterativennsimple.masked_linear_simplified_config <config.yaml>` prints
  it — as YAML, one line per block row, and as an aligned table — next to the
  shape and trainable-parameter count of the layer it builds.

The point of keeping both is RCP's, and it is the part I had missed: the two
formats fail differently, so **rendering one from the other is a check**. You write
the edge list, look at the grid, and a block you meant to write and didn't shows up
as a hole in a matrix rather than as an absent line in a list.

Limits, asserted in `tests/test_masked_linear_config.py`: grid → edges → grid is
exact; edges → grid → edges is exact up to the order of the blocks list, since the
edge list is a *set* of blocks; YAML comments do not survive any conversion, which
is why the command line prints to stdout and nothing writes a config file.

(Still unwritten, and probably not worth writing: the empty cell of the §2 table,
slice assignment with parse-to-types dispatch. It pairs the half of B I think is a
cost with the half of A I think is a cost.)

## 7. Verdict

**What was decided:** D was kept as `iterativennsimple/MaskedLinear_simplified.py`;
A, B and C were deleted; the original `MaskedLinear.py` stays as the reference
implementation and as the layer the existing config path uses. The configuration
language came back separately, in `masked_linear_simplified_config.py` (§6).

**What made the difference**, in RCP's words, given after he read the file and
before he saw any guess of mine:

1. On the test helper `train_briefly`: *"I like the name of the function. It is
   clear and approachable. That is a nice unit of code that I like and can lay the
   foundation for a student to copy that function and extend it."*
2. On the docstring's before-and-after example: *"I really like the way you gave the
   example and described it as 'longer on the page but shorter to read'. I think
   that single phrase captures a lot."*
3. On the `Block` type: *"the simple `Block` class is nice. It cuts at the joint."*
4. On `stack`: *"I found the `stack` function confusing at first glance. It is, in
   some sense, a '2D unzip' isn't it?"* — renamed `block_matrix`, with the unzip
   drawn in its docstring.
5. On the trainability functions and the "what torch does not provide" section:
   *"nice and cut at the joint."*
6. On what was missing: *"the one thing that I thirst for is what you called the
   'mini-language' that lets one define a MaskedLinear from a configuration."*

**What I had predicted, and where I was wrong.** §5 predicted the file would be
judged on 70 lines of code, flat nesting and the absence of an interpreter. Every
one of those is a property of the file *as a whole*, and not one of them is what
was actually cited. Four of the six items above are about **individual names** —
`train_briefly`, `Block`, `frozen`, `bernoulli`, `stack` — and the fifth is about a
sentence in a docstring. The unit of judgement was smaller than the unit I was
optimising, and "cuts at the joint" is a criterion I did not have: a name earns its
place when some operation consumes the thing it names whole, which is exactly what
separated `Block` (kept) from B's ten spec dataclasses (deleted).

I was right that the interpreter had to go, and wrong about why it mattered — not
because the file got shorter, but because a name you must read the body of is the
defect, and a mini-language is a hundred of them.

**Promoted to `PRINCIPLES.md` Part 1** as §1.5 (a name is a promise the body need
not be read), §1.6 (longer on the page can be shorter to read), §1.7 (introduce a
name only at a joint), §1.8 (write what the library does not provide, and mark the
boundary) and §1.9 (a configuration language belongs with the configuration files).

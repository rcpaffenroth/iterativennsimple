import pytest
import torch

from iterativennsimple.cudagraph import CUDAGraphFunctionCache


def test_cpu_fallback_matches_eager():
    def fn(x, y):
        return x + 2 * y

    runner = CUDAGraphFunctionCache(fn)
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)

    out = runner(x, y)

    assert torch.allclose(out, fn(x, y))
    assert runner.graph_count == 0
    assert runner.disabled_reason is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_cuda_graph_cache_reuses_matching_shapes():
    layer = torch.nn.Linear(8, 4).cuda()

    def fn(x, target):
        logits = layer(x)
        loss = torch.nn.functional.mse_loss(logits, target)
        return loss, logits

    runner = CUDAGraphFunctionCache(fn, name="test-step")
    x1 = torch.randn(16, 8, device="cuda")
    y1 = torch.randn(16, 4, device="cuda")
    x2 = torch.randn(16, 8, device="cuda")
    y2 = torch.randn(16, 4, device="cuda")

    loss1, logits1 = runner(x1, y1)
    loss2, logits2 = runner(x2, y2)

    assert runner.graph_count == 1
    assert loss1.is_cuda
    assert logits1.is_cuda
    assert loss2.is_cuda
    assert logits2.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_cuda_graph_cache_separates_batch_sizes():
    def fn(x):
        return x.square().sum(dim=1)

    runner = CUDAGraphFunctionCache(fn, name="shape-cache")

    out_a = runner(torch.randn(8, 4, device="cuda"))
    out_b = runner(torch.randn(12, 4, device="cuda"))

    assert out_a.shape == (8,)
    assert out_b.shape == (12,)
    assert runner.graph_count == 2

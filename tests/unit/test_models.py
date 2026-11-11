import torch

from industrial_visual_defect_detection.models import (
    PatchMemory,
    Student,
    Teacher,
    chunked_min_distance,
    distillation_loss,
    feature_maps,
)


def test_feature_shapes_and_distillation_gradient():
    teacher, student = Teacher(pretrained=False), Student()
    x = torch.rand(2, 3, 64, 64)
    t, s = teacher(x), student(x)
    assert [tuple(v.shape[-2:]) for v in t] == [(16, 16), (8, 8), (4, 4)]
    assert [tuple(v.shape[-2:]) for v in s] == [(16, 16), (8, 8), (4, 4)]
    loss, _ = distillation_loss(t, s)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert feature_maps(t, s, (64, 64)).shape == (2, 1, 64, 64)


def test_chunked_nearest_distances_match_hand_calculation():
    queries = torch.tensor([[[0., 0.], [3., 4.], [1., 0.]]])
    bank = torch.tensor([[0., 0.], [3., 0.]])
    result = chunked_min_distance(queries, bank, chunk=1)
    assert torch.allclose(result, torch.tensor([[0., 4., 1.]]))


def test_memory_fit_is_seeded_and_uses_only_passed_train_batches():
    class TinyFeatures(torch.nn.Module):
        def forward(self, x):
            value=x[:, :1].mean(dim=(2,3),keepdim=True)
            return value.expand(-1,64,-1,-1), value.expand(-1,128,-1,-1), value.expand(-1,256,-1,-1)

    def memory():
        model=PatchMemory.__new__(PatchMemory)
        torch.nn.Module.__init__(model)
        model.backbone=TinyFeatures()
        model.max_points=2
        model.register_buffer('bank',torch.empty(0,448))
        return model

    batches=[torch.full((1,3,4,4),.1),torch.full((1,3,4,4),.9)]
    first,second=memory(),memory()
    first.fit(batches,torch.device('cpu'),seed=9)
    second.fit(batches,torch.device('cpu'),seed=9)
    assert torch.equal(first.bank,second.bank)
    assert torch.allclose(first.bank[:,0].sort().values,torch.tensor([.1,.9]))

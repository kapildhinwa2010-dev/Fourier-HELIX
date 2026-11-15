import pytest
import torch
from fourier_helix import FourierHELIX, create_fourier_helix

class TestFourierHELIX:
    def test_init(self):
        m = FourierHELIX(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = FourierHELIX(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_fourier_helix()
        assert isinstance(m, FourierHELIX)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

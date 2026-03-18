"""Unit tests for FLUX.2 Klein KV cache classes and helpers."""

import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    Flux2KVCache,
    Flux2KVLayerCache,
    _blend_kv_mod_params,
)


class TestFlux2KVLayerCache(unittest.TestCase):
    def test_empty_cache(self):
        cache = Flux2KVLayerCache()
        k, v = cache.get()
        self.assertIsNone(k)
        self.assertIsNone(v)

    def test_store_and_get(self):
        cache = Flux2KVLayerCache()
        k = torch.randn(1, 64, 24, 128)
        v = torch.randn(1, 64, 24, 128)
        cache.store(k, v)

        k_out, v_out = cache.get()
        self.assertTrue(torch.equal(k_out, k))
        self.assertTrue(torch.equal(v_out, v))

    def test_store_overwrites(self):
        cache = Flux2KVLayerCache()
        cache.store(torch.randn(1, 32, 4, 64), torch.randn(1, 32, 4, 64))

        k2 = torch.randn(1, 16, 4, 64)
        v2 = torch.randn(1, 16, 4, 64)
        cache.store(k2, v2)

        k_out, _ = cache.get()
        self.assertEqual(k_out.shape[1], 16)

    def test_clear(self):
        cache = Flux2KVLayerCache()
        cache.store(torch.randn(1, 64, 24, 128), torch.randn(1, 64, 24, 128))
        cache.clear()

        k, v = cache.get()
        self.assertIsNone(k)
        self.assertIsNone(v)


class TestFlux2KVCache(unittest.TestCase):
    def test_layer_counts(self):
        cache = Flux2KVCache(num_double_layers=19, num_single_layers=38)
        self.assertEqual(len(cache.double_stream_caches), 19)
        self.assertEqual(len(cache.single_stream_caches), 38)

    def test_get_per_layer_cache(self):
        cache = Flux2KVCache(num_double_layers=3, num_single_layers=5)
        layer_cache = cache.get_double_cache(1)
        self.assertIsInstance(layer_cache, Flux2KVLayerCache)

        layer_cache = cache.get_single_cache(4)
        self.assertIsInstance(layer_cache, Flux2KVLayerCache)

    def test_per_layer_independence(self):
        cache = Flux2KVCache(num_double_layers=2, num_single_layers=2)
        cache.get_double_cache(0).store(
            torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64)
        )

        # Other layers should remain empty
        k, v = cache.get_double_cache(1).get()
        self.assertIsNone(k)
        k, v = cache.get_single_cache(0).get()
        self.assertIsNone(k)

    def test_clear_all(self):
        cache = Flux2KVCache(num_double_layers=2, num_single_layers=2)
        for i in range(2):
            cache.get_double_cache(i).store(
                torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64)
            )
            cache.get_single_cache(i).store(
                torch.randn(1, 8, 4, 64), torch.randn(1, 8, 4, 64)
            )

        cache.clear()
        for i in range(2):
            self.assertIsNone(cache.get_double_cache(i).get()[0])
            self.assertIsNone(cache.get_single_cache(i).get()[0])


class TestBlendKVModParams(unittest.TestCase):
    def test_output_shape(self):
        current = (
            torch.randn(2, 1, 256),
            torch.randn(2, 1, 256),
            torch.randn(2, 1, 256),
        )
        ref = (torch.randn(2, 1, 256), torch.randn(2, 1, 256), torch.randn(2, 1, 256))

        blended = _blend_kv_mod_params(current, ref, num_current=100, num_ref=32)

        self.assertEqual(len(blended), 3)
        for t in blended:
            self.assertEqual(t.shape, (2, 132, 256))

    def test_current_and_ref_regions(self):
        current = (torch.ones(1, 1, 4),)
        ref = (torch.zeros(1, 1, 4),)

        blended = _blend_kv_mod_params(current, ref, num_current=3, num_ref=2)

        # First 3 positions should be 1 (current), last 2 should be 0 (ref)
        self.assertTrue(torch.all(blended[0][:, :3, :] == 1.0))
        self.assertTrue(torch.all(blended[0][:, 3:, :] == 0.0))

    def test_batch_dimension_preserved(self):
        batch_size = 4
        current = (torch.randn(batch_size, 1, 128),)
        ref = (torch.randn(batch_size, 1, 128),)

        blended = _blend_kv_mod_params(current, ref, num_current=10, num_ref=5)
        self.assertEqual(blended[0].shape[0], batch_size)


if __name__ == "__main__":
    unittest.main()

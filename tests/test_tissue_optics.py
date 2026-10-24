"""
Unit tests for the tissue stack optics (simulation/tissue_stack.py).
"""

import unittest
from simulation.tissue_stack import BASE_LAYERS, eval_single_layer


class TestTissueOptics(unittest.TestCase):
    def test_evaluate_single_layer_tissue_and_air(self):
        # Normal chitin/melanin cuticle layer
        cuticle_layer = {
            'name': 'exocuticle_sclerotised',
            'thickness_um': 100.0,
            'composition': {'chitin': 0.6, 'melanin': 0.3, 'water': 0.1},
            'a_prime': 30.0,
            'f_rayleigh': 0.2,
            'b_mie': 1.1,
            'g500': 0.85
        }
        res_tissue = eval_single_layer(cuticle_layer, wl=532.0)
        self.assertGreater(res_tissue['mu_a'], 0.0)
        self.assertGreater(res_tissue['mu_s'], 0.0)
        self.assertGreaterEqual(res_tissue['g'], 0.0)
        self.assertLessEqual(res_tissue['g'], 1.0)
        self.assertGreater(res_tissue['n'], 1.0)
        self.assertFalse(res_tissue['is_air'])

        # Subelytral air layer
        air_layer = {
            'name': 'subelytral_air_gap',
            'thickness_um': 200.0,
            'composition': {'air': 1.0}
        }
        res_air = eval_single_layer(air_layer, wl=532.0)
        self.assertEqual(res_air['mu_a'], 0.0)
        self.assertEqual(res_air['mu_s'], 0.0)
        self.assertTrue(res_air['is_air'])
        self.assertEqual(res_air['n'], 1.0)

    def test_beetle_layers_evaluate(self):
        # Every beetle layer gives finite optics; only the air gap is transparent
        for lay in BASE_LAYERS:
            ev = eval_single_layer(lay, wl=532.0)
            self.assertEqual(ev['is_air'], lay['name'] == 'subelytral_air_gap')
            self.assertGreaterEqual(ev['mu_a'], 0.0)
            self.assertGreater(ev['rho_cp'], 0.0)


if __name__ == '__main__':
    unittest.main()

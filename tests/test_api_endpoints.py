"""
Unit tests for the LaserSim Flask API (server.py).
"""

import json
import unittest
from server import app


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'<html', res.data.lower())

    def test_setup_defaults(self):
        data = self.client.get('/api/setup').get_json()
        self.assertEqual(len(data['layers']), 11)
        self.assertEqual(data['laser']['wavelength_nm'], 532.0)
        self.assertIn('full', data['resolutions'])

    def test_evaluate_layers(self):
        layers = self.client.get('/api/setup').get_json()['layers']
        res = self.client.post('/api/layers/evaluate', data=json.dumps({'layers': layers, 'wavelength_nm': 1470}),
                               content_type='application/json')
        out = res.get_json()['layers']
        self.assertEqual(len(out), len(layers))
        self.assertEqual(out[3]['mu_a'], 0.0)                 # air gap
        self.assertGreater(out[6]['mu_a'], out[0]['mu_a'])    # hemolymph water absorbs at 1470 nm
        self.assertEqual(out[-1]['z_end_um'], sum(l['thickness_um'] for l in layers))

    def test_run_streams_setup_frames_and_done(self):
        config = {'laser': {'resolution': 'preview', 'duration_ms': 6, 'pulse_ms': 4, 'photons': 5000}}
        res = self.client.get('/api/run?config=' + json.dumps(config))
        self.assertIn('text/event-stream', res.content_type)
        events = [json.loads(chunk[6:]) for chunk in res.get_data(as_text=True).strip().split('\n\n')]
        kinds = [e['event'] for e in events]
        self.assertEqual(kinds[0], 'setup')
        self.assertEqual(kinds[-1], 'done')
        frames = [e['data'] for e in events if e['event'] == 'frame']
        self.assertGreater(len(frames), 1)
        self.assertGreater(frames[-1]['energy_j']['absorbed'], 0.0)
        self.assertFalse(frames[-1]['laser_on'])


if __name__ == '__main__':
    unittest.main()

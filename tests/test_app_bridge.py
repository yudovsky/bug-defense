"""
Unit tests for app bridge and networking utilities (app.py).
"""

import unittest
import socket
from app import get_free_port, is_port_in_use, LaserSimBridge


class TestAppBridge(unittest.TestCase):
    def test_get_free_port_returns_valid_port(self):
        port = get_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 1024)
        self.assertLess(port, 65536)

    def test_is_port_in_use_on_bound_socket(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        s.listen(1)
        bound_port = s.getsockname()[1]
        try:
            self.assertTrue(is_port_in_use(bound_port))
        finally:
            s.close()
            # Once closed, should either return False or become quickly available
            self.assertFalse(is_port_in_use(bound_port))

    def test_laser_sim_bridge_methods(self):
        bridge = LaserSimBridge(backend_port=5050, debug=True)
        self.assertEqual(bridge.get_backend_port(), 5050)
        self.assertTrue(bridge.is_debug_mode())

        bridge_prod = LaserSimBridge(backend_port=8080, debug=False)
        self.assertEqual(bridge_prod.get_backend_port(), 8080)
        self.assertFalse(bridge_prod.is_debug_mode())


if __name__ == '__main__':
    unittest.main()

import argparse
import os
import socket
import threading
import time
import sys

# Configure macOS Cocoa application process and bundle name so the OS menu bar
# displays the native application name ("LaserSim") rather than "Python".
try:
    import Foundation, AppKit
    bundle = AppKit.NSBundle.mainBundle()
    info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
    if info is not None:
        info['CFBundleName'] = 'LaserSim'
        info['CFBundleDisplayName'] = 'LaserSim'
    Foundation.NSProcessInfo.processInfo().setProcessName_('LaserSim')
except Exception:
    pass

import webview
try:
    from webview.menu import Menu, MenuAction, MenuSeparator
except ImportError:
    Menu, MenuAction, MenuSeparator = None, None, None

# Import the multi-physics Flask app
from server import app as flask_app

# Disable default browser menus for clean desktop look
webview.settings['SHOW_DEFAULT_MENUS'] = False

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

class LaserSimBridge:
    def __init__(self, backend_port: int, debug: bool = False):
        self.backend_port = backend_port
        self.debug = debug
        
    def get_backend_port(self):
        return self.backend_port

    def is_debug_mode(self):
        return self.debug

def build_app_menus(window, app_url=None):
    def run_js(code):
        def _exec():
            try:
                window.evaluate_js(code)
            except Exception as e:
                print(f"[NativeMenu] Error: {e}")
        threading.Thread(target=_exec, daemon=True).start()

    def reload_app():
        if app_url:
            window.load_url(app_url)
        else:
            run_js("window.location.reload()")

    return [
        Menu("File", [
            MenuAction("Reload App", reload_app),
            MenuSeparator(),
            MenuAction("Close Window", lambda: window.destroy()),
        ]),
        Menu("Pages", [
            MenuAction("1. Beetle & Laser", lambda: run_js("showPage('setup')")),
            MenuAction("2. Simulation", lambda: run_js("showPage('sim')")),
        ]),
        Menu("Simulation", [
            MenuAction("Run Simulation", lambda: run_js("startRun()")),
            MenuAction("Stop Simulation", lambda: run_js("stopRun()")),
        ]),
        Menu("Wavelength Presets", [
            MenuAction(label, (lambda wl: lambda: run_js(f"document.getElementById('wavelength').value={wl}; onLaserChange();"))(wl))
            for wl, label in [
                (320, "320 nm (UV - Melanin and Protein)"),
                (450, "450 nm (Blue Diode)"),
                (532, "532 nm (Green)"),
                (1064, "1064 nm (Nd:YAG - Deep Scattering)"),
                (1470, "1470 nm (Diode - Water)"),
                (1940, "1940 nm (Thulium - Strong Water)"),
                (10600, "10600 nm (CO2 - Surface)"),
            ]
        ]),
    ]

def run_flask(host: str, port: int):
    # Run Flask with threading enabled and minimal logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    flask_app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LaserSim Desktop App")
    parser.add_argument("--server-only", action="store_true", help="Run HTTP server without native GUI window")
    parser.add_argument("--port", type=int, default=5050, help="Port to run server on (default: 5050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable developer tools and WebKit inspection")
    args = parser.parse_args()

    port = args.port
    # If default port 5050 is already occupied by a background server, check or use a free port
    if is_port_in_use(port) and not args.server_only:
        # Check if it's our own Flask app
        try:
            import urllib.request
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            resp = opener.open(f"http://{args.host}:{port}/", timeout=0.8)
            server_already_running = True
        except Exception:
            port = get_free_port()
            server_already_running = False
    else:
        server_already_running = False

    debug_mode = bool(args.debug or os.environ.get("LASERSIM_DEBUG", "").lower() in ("1", "true", "yes"))

    if args.server_only:
        print(f"Starting LaserSim multi-physics server in headless mode on http://{args.host}:{port}/")
        run_flask(args.host, port)
    else:
        if not server_already_running:
            # Start Flask in background thread
            t = threading.Thread(target=run_flask, args=(args.host, port), daemon=True)
            t.start()
            time.sleep(0.4)

        bridge = LaserSimBridge(port, debug=debug_mode)
        app_url = f"http://{args.host}:{port}/"
        
        window = webview.create_window(
            title="LaserSim — Beetle Laser Ablation",
            url=app_url,
            js_api=bridge,
            width=1480,
            height=920,
            background_color="#121214",
            min_size=(1050, 700)
        )
        menus = build_app_menus(window, app_url) if Menu else []
        print(f"Launching LaserSim desktop application window connected to {app_url}...")
        webview.start(debug=debug_mode, menu=menus)

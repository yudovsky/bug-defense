"""
LaserSim backend (Flask).
- /api/setup: default beetle layers and laser settings
- /api/layers/evaluate: optical properties of each layer at a wavelength
- /api/run: runs the coupled 3D simulation and streams the four-panel frames as Server-Sent Events
- /api/stop: stops the running simulation
"""

import json
import threading
import traceback
from flask import Flask, request, jsonify, render_template, Response, stream_with_context

from simulation.beetle_sim import default_setup, evaluate_layers, run

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# One simulation at a time: the engine holds the GPU and several GB of state at full resolution
run_lock = threading.Lock()
stop_flag = threading.Event()


def sse(event, data):
    return f"data: {json.dumps({'event': event, 'data': data})}\n\n"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/setup', methods=['GET'])
def api_setup():
    return jsonify(default_setup())


@app.route('/api/layers/evaluate', methods=['POST'])
def api_evaluate_layers():
    body = request.get_json(force=True) or {}
    try:
        layers = evaluate_layers(body.get('layers', []), float(body.get('wavelength_nm', 532.0)))
        return jsonify({'status': 'success', 'layers': layers})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/stop', methods=['POST'])
def api_stop():
    stop_flag.set()
    return jsonify({'status': 'success'})


@app.route('/api/run', methods=['GET'])
def api_run():
    config = json.loads(request.args.get('config', '{}'))
    layers = config.get('layers') or default_setup()['layers']
    laser = dict(default_setup()['laser'], **config.get('laser', {}))

    def stream():
        # A previous run that was just stopped may still be finishing its current step
        stop_flag.set()
        if not run_lock.acquire(timeout=30):
            yield sse('error', {'message': 'Another simulation is still running.'})
            return
        stop_flag.clear()
        try:
            for ev in run(layers, laser):
                if stop_flag.is_set():
                    return
                yield sse(ev['event'], ev['data'])
        except Exception as e:
            traceback.print_exc()
            yield sse('error', {'message': str(e)})
        finally:
            run_lock.release()

    return Response(stream_with_context(stream()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    print("Starting LaserSim server on port 5050...")
    app.run(host='127.0.0.1', port=5050, debug=False, threaded=True)

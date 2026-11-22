import json
import time

import numpy as np
import pytest

from klongpy import KlongInterpreter
from klongpy.backends import get_backend, list_backends
from .runtime import evaluate, load


pytestmark = pytest.mark.v2_gpu


@pytest.mark.parametrize(("expression", "answer"), [
    ("energy(v)", 163840.0),
    ("(energy:>[2.0 -3.0 0.5])~[4.0 -6.0 1.0]", 1),
])
def test_native_gpu_result(kg, gpu_device, tmp_path, expression, answer):
    output = evaluate(kg, f'{load("gpu.kg")};{expression}', tmp_path,
                      options=["--device", gpu_device], timeout=45)
    assert float(output) == answer


def test_native_device_residency(kg, gpu_device, tmp_path, record_property):
    # Diagnostic contract for this one expression. Fused kernels also satisfy it.
    trace = tmp_path / "device.json"
    start = time.perf_counter()
    result = evaluate(kg, f'{load("gpu.kg")};energy(v)', tmp_path, options=["--device", gpu_device],
                      extra_env={"KLONG_GPU_TRACE": str(trace)}, timeout=45)
    record_property("gpu_end_to_end_seconds", time.perf_counter() - start)
    assert float(result) == 163840.0
    assert trace.is_file(), "KLONG_GPU_TRACE must report the actual device and data transfers"
    stats = json.loads(trace.read_text())
    assert stats["device"].split(":")[0] == gpu_device.split(":")[0]
    if ":" in gpu_device:
        assert stats["device"] == gpu_device
    assert stats["kernel_count"] >= 1
    assert {"*", "+/"}.issubset(stats["device_operations"]), stats
    assert 0 <= stats["host_readback_bytes"] <= 8, "Only the final scalar needs to return"


def device_interpreter(gpu_device):
    kind = gpu_device.split(":")[0]
    backends = [name for name in list_backends()
                if any(d.split(":")[0] == kind for d in get_backend(name).list_devices())]
    assert backends, f"No Python array backend exposes {gpu_device}"
    return KlongInterpreter(backend=backends[0], device=gpu_device)


@pytest.mark.v2_hosted_gpu
def test_python_tensors_on_selected_device(gpu_device):
    kind = gpu_device.split(":")[0]
    kg = device_interpreter(gpu_device)
    kg(load("gpu.kg"))
    batch = kg("square(v)")
    assert str(batch.device).split(":")[0] == kind
    if ":" in gpu_device:
        assert str(batch.device) == gpu_device
    kg["batch"] = batch
    assert float(kg("+/batch")) == 163840.0
    gradient = kg("energy:>[2.0 -3.0 0.5]")
    assert str(gradient.device) == str(batch.device)
    np.testing.assert_allclose(kg.backend.to_numpy(gradient), [4.0, -6.0, 1.0], rtol=1e-6)


@pytest.mark.v2_hosted_gpu
@pytest.mark.parametrize("expression", ["[1.5 2.5]∇cubeSum", ".grad(cubeSum;[1.5 2.5])"])
def test_numeric_gradient_returns_to_device(gpu_device, expression):
    interpreter = device_interpreter(gpu_device)
    interpreter(load("gpu.kg"))
    gradient = interpreter(expression)
    assert interpreter.backend.is_backend_array(gradient)
    assert str(gradient.device) == str(interpreter("square([1.5 2.5])").device)
    np.testing.assert_allclose(interpreter.backend.to_numpy(gradient), [6.75, 18.75], rtol=1e-5)


def test_native_query_feeds_gpu(kg, gpu_device, tmp_path):
    expr = f'{load("market.kg")};v::lastpx();+/v*v'
    trace = tmp_path / "query-device.json"
    output = evaluate(kg, expr, tmp_path, options=["--device", gpu_device],
                      extra_env={"KLONG_GPU_TRACE": str(trace)}, timeout=45)
    assert float(output) == 103**2 + 201**2 + 300**2
    stats = json.loads(trace.read_text())
    assert {"*", "+/"}.issubset(stats["device_operations"]), stats


@pytest.mark.parametrize("expression,expected", [
    pytest.param("((!257)!7)*2.0", (np.arange(257) % 7) * 2.0, id="partial-final-block"),
    pytest.param("2.0+[[1.0 -2.0] [3.0 4.0]]", [[3., 0.], [5., 6.]], id="matrix-broadcast"),
    pytest.param("square([-1.5 0.0 0.5 4.0])", [2.25, 0., 0.25, 16.], id="fractional-and-zero"),
    pytest.param("energy:>[0.0 0.0]", [0., 0.], id="zero-gradient-vector"),
    pytest.param("energy:>[-0.5]", [-1.], id="singleton-gradient"),
])
@pytest.mark.parametrize("kind", ["kg", pytest.param("kgpy", marks=pytest.mark.v2_hosted_gpu)])
def test_device_arrays(kind, expression, expected, kg, gpu_device, tmp_path):
    if kind == "kg":
        output = evaluate(kg, f'{load("gpu.kg")};.p(.json({expression}));1', tmp_path,
                          options=["--device", gpu_device], timeout=45)
        actual = json.loads(output.splitlines()[0])
    else:
        interp = device_interpreter(gpu_device)
        interp(load("gpu.kg"))
        tensor = interp(expression)
        assert str(tensor.device).split(":")[0] == gpu_device.split(":")[0]
        if ":" in gpu_device:
            assert str(tensor.device) == gpu_device
        actual = interp.backend.to_numpy(tensor)
    assert np.asarray(actual).shape == np.asarray(expected).shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import main


class InferenceBackendTests(unittest.TestCase):
    def test_progress_sidecar_replace_retries_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as directory:
            progress_path = Path(directory) / "progress.json"
            real_replace = os.replace
            attempts = 0

            def temporarily_locked_replace(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("temporarily locked")
                return real_replace(source, destination)

            with patch(
                "main.os.replace",
                side_effect=temporarily_locked_replace,
            ) as replace:
                main._write_progress_sidecar(progress_path, 100, 1000, "running")

            self.assertEqual(replace.call_count, 3)
            self.assertEqual(
                progress_path.read_text(encoding="utf-8"),
                '{"schema_version":"1.0","status":"running",'
                '"frames_processed":100,"total_frames":1000}',
            )
            self.assertEqual(list(Path(directory).iterdir()), [progress_path])

    @patch("main.importlib.util.find_spec", return_value=None)
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_uses_cpu_when_cuda_and_openvino_are_unavailable(
        self, _cuda_available, _openvino_spec
    ):
        self.assertEqual(
            main._resolve_inference_backend("auto", None),
            ("cpu", "cpu"),
        )
        _cuda_available.assert_called_once_with()
        _openvino_spec.assert_called_once_with("openvino")

    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=False)
    def test_auto_prefers_openvino_over_torch_cpu(
        self, _cuda_available, _openvino_spec
    ):
        self.assertEqual(
            main._resolve_inference_backend("auto", None),
            ("openvino", "cpu"),
        )
        _cuda_available.assert_called_once_with()
        _openvino_spec.assert_called_once_with("openvino")

    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=False)
    def test_explicit_cpu_device_prefers_openvino_if_installed(
        self, _cuda_available, _openvino_spec
    ):
        self.assertEqual(
            main._resolve_inference_backend("auto", "cpu"),
            ("openvino", "cpu"),
        )
        _cuda_available.assert_called_once_with()
        _openvino_spec.assert_called_once_with("openvino")

    @patch("openvino.Core")
    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=False)
    def test_openvino_resolves_intel_gpu_device(
        self, _cuda_available, _openvino_spec, core
    ):
        core.return_value.available_devices = ["CPU", "GPU"]

        self.assertEqual(
            main._resolve_inference_backend("openvino", "intel:gpu"),
            ("openvino", "intel:gpu"),
        )

    @patch("openvino.Core")
    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=True)
    def test_auto_honors_explicit_intel_device_over_cuda(
        self, _cuda_available, _openvino_spec, core
    ):
        core.return_value.available_devices = ["CPU", "GPU"]

        self.assertEqual(
            main._resolve_inference_backend("auto", "intel:gpu"),
            ("openvino", "intel:gpu"),
        )

    @patch("openvino.Core")
    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=False)
    def test_openvino_rejects_unavailable_intel_device(
        self, _cuda_available, _openvino_spec, core
    ):
        core.return_value.available_devices = ["CPU"]

        with self.assertRaisesRegex(RuntimeError, "device 'GPU' is unavailable"):
            main._resolve_inference_backend("openvino", "intel:gpu")

    def test_cli_accepts_explicit_openvino_intel_gpu(self):
        args = main._parse_args(
            ["--backend", "openvino", "--device", "intel:gpu"]
        )

        self.assertEqual(args.backend, "openvino")
        self.assertEqual(args.device, "intel:gpu")

    def test_cli_defaults_to_benchmarked_inference_size(self):
        with patch.dict(os.environ, {}, clear=True):
            args = main._parse_args([])

        self.assertEqual(args.imgsz, 384)

    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_auto_prefers_cuda_over_openvino(
        self, _device_count, _cuda_available, _openvino_spec
    ):
        self.assertEqual(
            main._resolve_inference_backend("auto", None),
            ("cuda", 0),
        )
        _cuda_available.assert_called_once_with()
        _device_count.assert_called_once_with()
        _openvino_spec.assert_called_once_with("openvino")

    @patch("torch.cuda.device_count", return_value=1)
    @patch("torch.cuda.is_available", return_value=True)
    def test_explicit_cuda_device_index_is_checked(
        self, _cuda_available, _device_count
    ):
        self.assertEqual(
            main._resolve_inference_backend("auto", "cuda:0"),
            ("cuda", 0),
        )
        with self.assertRaisesRegex(ValueError, "index 1 is unavailable"):
            main._resolve_inference_backend("cuda", "1")

    @patch("main.importlib.util.find_spec", return_value=None)
    @patch("torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_fails_when_unavailable(
        self, _cuda_available, _openvino_spec
    ):
        with self.assertRaisesRegex(RuntimeError, "CUDA was requested"):
            main._resolve_inference_backend("cuda", None)
        _cuda_available.assert_called_once_with()
        _openvino_spec.assert_called_once_with("openvino")

    @patch("torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("torch.cuda.get_device_name", return_value="Test NVIDIA GPU")
    def test_cuda_probe_runs_the_selected_model_on_a_small_input(
        self, _device_name, _device_capability
    ):
        model = Mock()
        frame = object()

        main._probe_cuda_inference(model, frame, 512)

        model.predict.assert_called_once_with(
            source=frame,
            device=0,
            imgsz=320,
            verbose=False,
            save=False,
        )
        _device_name.assert_called_once_with(0)
        _device_capability.assert_called_once_with(0)

    @patch("torch.cuda.get_device_capability", return_value=(12, 0))
    @patch("torch.cuda.get_device_name", return_value="Test NVIDIA GPU")
    def test_cuda_probe_failure_reports_gpu_and_build_details(
        self, _device_name, _device_capability
    ):
        model = Mock()
        model.predict.side_effect = RuntimeError("unsupported CUDA kernel")

        with self.assertRaisesRegex(
            RuntimeError,
            "Test NVIDIA GPU.*capability \\(12, 0\\).*unsupported CUDA kernel",
        ):
            main._probe_cuda_inference(model, object(), 512)
        _device_name.assert_called_once_with(0)
        _device_capability.assert_called_once_with(0)

    @patch("main.importlib.util.find_spec", return_value=object())
    @patch("torch.cuda.is_available", return_value=False)
    def test_openvino_export_is_cached_for_the_source_model(
        self, _cuda_available, _openvino_spec
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "best.pt"
            model_path.touch()
            exported_directory = Path(temporary_directory) / "best_openvino_model"
            exported_directory.mkdir()
            exported_xml = exported_directory / "best.xml"
            export_calls = []

            class FakeYOLO:
                def __init__(self, path):
                    self.path = path
                    self.names = {0: "player"}

                def export(self, **_kwargs):
                    export_calls.append((self.path, _kwargs))
                    exported_xml.touch()
                    (exported_directory / "metadata.yaml").write_text(
                        f"imgsz:\n- {_kwargs['imgsz']}\n- {_kwargs['imgsz']}\n",
                        encoding="utf-8",
                    )
                    return str(exported_directory)

            with patch("main.YOLO", side_effect=FakeYOLO) as yolo_factory:
                model, device, backend = main._load_inference_model(
                    model_path, "openvino", None, 512
                )
                self.assertEqual(model.path, str(exported_directory))
                self.assertEqual(device, "cpu")
                self.assertEqual(backend, "openvino")
                self.assertEqual(
                    yolo_factory.call_args_list,
                    [
                        call(str(model_path)),
                        call(str(exported_directory)),
                    ],
                )

                model, device, backend = main._load_inference_model(
                    model_path, "openvino", None, 512
                )

                self.assertEqual(backend, "openvino")
                self.assertEqual(device, "cpu")
                self.assertEqual(
                    export_calls,
                    [
                        (
                            str(model_path),
                            {
                                "format": "openvino",
                                "imgsz": 512,
                                "device": "cpu",
                            },
                        )
                    ],
                )
                self.assertEqual(yolo_factory.call_count, 3)

                main._load_inference_model(model_path, "openvino", None, 640)
                self.assertEqual(
                    [call_args[1]["imgsz"] for call_args in export_calls],
                    [512, 640],
                )
        self.assertEqual(_cuda_available.call_count, 3)
        self.assertEqual(_openvino_spec.call_count, 3)


if __name__ == "__main__":
    unittest.main()

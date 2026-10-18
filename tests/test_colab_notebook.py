import ast
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_ollama.ipynb"


def cell_source(cell_id):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "".join(next(cell for cell in cells if cell["id"] == cell_id)["source"])


def command_runner():
    tree = ast.parse(cell_source("gpu-check"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["run_command"]


def test_command_output_and_stderr_are_visible_in_notebook(capsys):
    run = command_runner()
    run([sys.executable, "-c", "import sys; print('stdout', flush=True); print('stderr', file=sys.stderr)"])
    printed = capsys.readouterr().out
    assert "stdout" in printed
    assert "stderr" in printed


def test_command_error_prints_actual_cause_before_exception(capsys):
    run = command_runner()
    with pytest.raises(RuntimeError, match="7"):
        run([sys.executable, "-c", "import sys; print('installation failed', file=sys.stderr); sys.exit(7)"])
    assert "installation failed" in capsys.readouterr().out


@pytest.mark.parametrize("zstd_installed", [False, True])
def test_installation_checks_zstd_and_separates_download_from_execution(zstd_installed):
    calls = []
    namespace = {
        "run_command": lambda command, **kwargs: calls.append(command),
        "shutil": SimpleNamespace(which=lambda name: "/usr/bin/zstd" if zstd_installed else None),
        "os": SimpleNamespace(geteuid=lambda: 0),
    }
    exec(compile(cell_source("install-ollama"), str(NOTEBOOK), "exec"), namespace)
    assert any(command[:3] == ["apt-get", "install", "-y"] and "zstd" in command
               for command in calls) is not zstd_installed
    assert calls[-2][0] == "curl"
    assert calls[-1] == ["sh", "/tmp/jev-ollama-install.sh"]


def test_notebook_launch_respects_selected_case_limit():
    calls = []
    namespace = {
        "UV_BIN": "uv", "PROJECT_DIR": Path("/content"),
        "run_command": lambda command, **kwargs: calls.append((command, kwargs)),
    }
    exec(compile(cell_source("run-pipeline"), str(NOTEBOOK), "exec"), namespace)
    command, kwargs = calls[0]
    assert "--debug" in command
    assert command[command.index("--split") + 1] == "dev"
    if namespace["CASE_LIMIT"] is None:
        assert "--limit" not in command
    else:
        assert command[command.index("--limit") + 1] == str(namespace["CASE_LIMIT"])
    assert kwargs["cwd"] == namespace["PROJECT_DIR"]


def test_notebook_can_display_a_run_without_successful_cases(tmp_path, capsys):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "predictions.jsonl").write_text("")
    (outputs / "predictions.errors.jsonl").write_text(
        json.dumps({"case_id": "C1", "error": "attempts exhausted"}) + "\n",
    )
    namespace = {"PROJECT_DIR": tmp_path, "json": json}
    exec(compile(cell_source("read-output"), str(NOTEBOOK), "exec"), namespace)
    printed = capsys.readouterr().out
    assert "Нет успешных результатов" in printed
    assert "C1" in printed and "attempts exhausted" in printed


def test_notebook_explicitly_configures_schema_and_temperature():
    namespace = {"os": SimpleNamespace(environ={}), "MODEL": "small", "OLLAMA_URL": "http://127.0.0.1:11434"}
    first_statement = ast.parse(cell_source("configure")).body[0]
    exec(compile(ast.Module(body=[first_statement], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    configured = namespace["os"].environ
    assert configured["LLM_API_PROTOCOL"] == "openai"
    assert configured["LLM_MIN_REQUEST_INTERVAL_SECONDS"] == "0"
    assert configured["LLM_RATE_LIMIT_RETRIES"] == "1"
    assert configured["LLM_RESPONSE_FORMAT"] == "json_schema"
    assert configured["LLM_TEMPERATURE"] == "0"


def test_notebook_preflight_checks_schema_api_without_network():
    requests = []

    def urlopen(request, **kwargs):
        requests.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{
            "finish_reason": "stop", "message": {"content": '{"ready":true}'},
        }]}).encode())

    namespace = {
        "run_command": lambda *args: None, "MODEL": "small", "OLLAMA_URL": "http://127.0.0.1:11434",
        "json": json, "Request": Request, "urlopen": urlopen,
    }
    exec(compile(cell_source("model-check"), str(NOTEBOOK), "exec"), namespace)
    output = requests[0]["response_format"]
    assert output["type"] == "json_schema"
    assert output["json_schema"]["schema"]["properties"]["ready"] == {"type": "boolean", "const": True}

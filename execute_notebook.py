"""현재 Python 환경으로 제출 노트북을 전체 실행하고 결과를 저장한다."""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    """가상환경과 같은 인터프리터의 커널로 모든 셀을 순서대로 실행한다."""
    root = Path(__file__).resolve().parent
    # 커널 캐시는 프로젝트 임시 폴더에 두어 사용자 전역 환경을 변경하지 않는다.
    (root / "tmp/ipython").mkdir(parents=True, exist_ok=True)
    os.environ["IPYTHONDIR"] = str(root / "tmp/ipython")
    path = root / "suwon_bus_demand_ml.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # 시스템 Python 대신 설치가 완료된 현재 가상환경을 사용해 패키지 불일치를 막는다.
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=300, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        # 실패한 셀도 보존하여 오류를 숨기지 않고 확인할 수 있게 한다.
        nbformat.write(notebook, path)
    print(f"전체 실행 완료: {path.name}")


if __name__ == "__main__":
    main()

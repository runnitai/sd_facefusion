import os
import subprocess
import sys
import traceback
import urllib.request
from importlib import metadata
from typing import Optional

from packaging import version as pv
from tqdm import tqdm

try:
    from modules.paths_internal import models_path, script_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")


req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models_dir = os.path.join(models_path, "facefusion")


def pip_install(*args):
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "install"] + list(args),
        stderr=subprocess.STDOUT,
    )
    for line in output.decode().split("\n"):
        if "Successfully installed" in line:
            print(line)


def is_installed(pkg: str, version: Optional[str] = None, check_strict: bool = True) -> bool:
    try:
        # Retrieve the package version from the installed package metadata
        installed_version = metadata.version(pkg)
        print(f"Installed version of {pkg}: {installed_version}")
        # If version is not specified, just return True as the package is installed
        if version is None:
            return True

        # Compare the installed version with the required version
        if check_strict:
            # Strict comparison (must be an exact match)
            return pv.parse(installed_version) == pv.parse(version)
        else:
            # Non-strict comparison (installed version must be greater than or equal to the required version)
            return pv.parse(installed_version) >= pv.parse(version)

    except metadata.PackageNotFoundError:
        # The package is not installed
        return False
    except Exception as e:
        # Any other exceptions encountered
        print(f"Error: {e}")
        return False


def download(url, path):
    request = urllib.request.urlopen(url)
    total = int(request.headers.get('Content-Length', 0))
    with tqdm(total=total, desc='Downloading...', unit='B', unit_scale=True, unit_divisor=1024) as progress:
        urllib.request.urlretrieve(url, path,
                                   reporthook=lambda count, block_size, total_size: progress.update(block_size))


if not os.path.exists(models_dir):
    os.makedirs(models_dir)


def install_requirements():
    req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")
    req_file_startup_arg = os.environ.get("REQS_FILE", "requirements_versions.txt")

    if req_file == req_file_startup_arg:
        return
    print("Checking Facefusion requirements...")
    non_strict_separators = ["==", ">=", "<=", ">", "<", "~="]
    # Load the requirements file
    with open(req_file, "r") as f:
        reqs = f.readlines()

    for line in reqs:
        try:
            package = line.strip()
            if package and not package.startswith("#"):
                package_version = None
                strict = "==" in package
                for separator in non_strict_separators:
                    if separator in package:
                        strict = separator == "=="
                        parts = line.split(separator)
                        if len(parts) < 2:
                            print(f"Invalid requirement: {line}")
                            continue
                        package = parts[0].strip()
                        package_version = parts[1].strip()
                        if "#" in package_version:
                            package_version = package_version.split("#")[0]
                        package = package.strip()
                        package_version = package_version.strip()
                        break
                if "#" in package:
                    package = package.split("#")[0]
                package = package.strip()
                v_string = "" if not package_version else f" v{package_version}"
                if not is_installed(package, package_version, strict):
                    print(f"[Facefusion] {package}{v_string} is not installed.")
                    pip_install(line)
                else:
                    print(f"[Facefusion] {package}{v_string} is already installed.")

        except subprocess.CalledProcessError as grepexc:
            error_msg = grepexc.stdout.decode()
            print_requirement_installation_error(error_msg)


def print_requirement_installation_error(err):
    print("# Requirement installation exception:")
    for line in err.split('\n'):
        line = line.strip()
        if line:
            print(line)


def install_runtimes():
    torch_version = '2.0.1'
    torch_cuda_wheel = 'cu118'  # Update this to the correct CUDA version if needed
    onnxruntime_version = '1.16.3'
    onnxruntime_cuda_name = 'onnxruntime-gpu'
    # Uninstall existing PyTorch and ONNX Runtime installations
    if not is_installed(onnxruntime_cuda_name, onnxruntime_version, True):
        print(f"Installing {onnxruntime_cuda_name}...")
        subprocess.call(['pip', 'uninstall', 'onnxruntime', onnxruntime_cuda_name, '-y'])
        pip_install(f"{onnxruntime_cuda_name}=={onnxruntime_version}")
    if not is_installed('torch', torch_version, True):
        print(f"Installing torch...")
        # Install the specified version of PyTorch with CUDA support
        pip_install(f'torch=={torch_version}+{torch_cuda_wheel}', '--extra-index-url',
                    'https://download.pytorch.org/whl/' + torch_cuda_wheel)


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ext_dir = os.path.join(base_dir, 'extensions-builtin', 'sd_facefusion', 'facefusion')

if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)
else:
    print(f"Ext dir already in path: {ext_dir}")

install_requirements()
install_runtimes()

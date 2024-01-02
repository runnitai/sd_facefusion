import platform
import subprocess
import os, sys
from typing import Optional, Dict, Tuple
from packaging import version as pv

from importlib import metadata
from tqdm import tqdm
import urllib.request
from AutoIntegrityCheck.validator import validate_auto_integrity

try:
    from modules.paths_internal import models_path, script_path
except:
    try:
        from modules.paths import models_path
    except:
        model_path = os.path.abspath("models")

if not validate_auto_integrity(script_path):
    print("AutoIntegrityCheck failed, exiting...")
    exit(1)

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


def is_installed(pkg: str, version: Optional[str] = None, separator: str = None) -> bool:
    try:
        # Retrieve the package version from the installed package metadata
        installed_version = metadata.version(pkg)

        # If version is not specified, just return True as the package is installed
        if version is None:
            return True

        # Compare the installed version with the required version
        if separator == "==":
            # Strict comparison (must be an exact match)
            return pv.parse(installed_version) == pv.parse(version)
        elif ">" in separator:
            # Non-strict comparison (installed version must be greater than or equal to the required version)
            return pv.parse(installed_version) >= pv.parse(version)
        elif "<" in separator:
            # Non-strict comparison (installed version must be less than or equal to the required version)
            return pv.parse(installed_version) <= pv.parse(version)
        elif "~=" in separator:
            # Non-strict comparison (installed version must be within the same minor version)
            return pv.parse(installed_version).base_version == pv.parse(version).base_version
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
    print("Checking FaceFusion requirements...")
    separators = ["==", ">=", "<=", ">", "<", "~="]

    with open(req_file) as file:
        requirements = file.readlines()

    for package in requirements:
        package_version = None
        try:
            package = package.strip()
            strict = False
            package_separator = None
            for separator in separators:
                if separator in package:
                    strict = True
                    package, package_version = package.split(separator)
                    package = package.strip()
                    package_version = package_version.strip()
                    package_separator = separator
                    break
            if not is_installed(package, package_version, package_separator):
                print(f"[FaceFusion] Installing {package}...")
                pip_install(package)
            else:
                print(f"[FaceFusion] {package} is already installed")
        except Exception as e:
            print(e)
            print(f"\nERROR: Failed to install {package} - ReActor won't start")
            raise e


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

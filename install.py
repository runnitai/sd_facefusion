import os
import subprocess
import sys
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


# models_dir = os.path.join(models_path, "facefusion")


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
        if version is None and installed_version is not None:
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
    onnxruntime_version = '1.20.1'
    onnxruntime_cuda_name = 'onnxruntime-gpu'
    # Uninstall existing PyTorch and ONNX Runtime installations
    if not is_installed(onnxruntime_cuda_name, onnxruntime_version, True):
        print(f"Installing {onnxruntime_cuda_name}...")
        subprocess.call(['pip', 'uninstall', 'onnxruntime', onnxruntime_cuda_name, '-y'])
        pip_install(f"{onnxruntime_cuda_name}=={onnxruntime_version}")


def install_torchaudio():
    cu_version = 'cu118'
    # Do some package magick and get the version of torch installed, including the cuda version
    torch_version = metadata.version('torch')
    if torch_version is None:
        print("Error: torch is not installed.")
        return
    if '+' in torch_version:
        torch_version_parts = torch_version.split('+')
        torch_pkg_version = torch_version_parts[0]
        cu_version = torch_version_parts[1]
    else:
        torch_pkg_version = torch_version
    # Set index_url to the correct URL for the installed torch version
    index_url = f"https://download.pytorch.org/whl/{cu_version}"
    # Install torchvision
    if not is_installed('torchaudio'):
        print("Installing torchaudio...")
        pip_install(f'torchaudio=={torch_pkg_version}', f'--index-url={index_url}')


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ext_dir = os.path.join(base_dir, 'extensions', 'sd_facefusion')

if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)
else:
    print(f"Ext dir already in path: {ext_dir}")

install_requirements()
install_runtimes()
install_torchaudio()

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import packaging.markers
import packaging.version
from packaging.requirements import Requirement
from packaging.version import Version


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def get_project_from_setup_py(path: str) -> dict[str, Any] | None:
    import ast

    try:
        with open(os.path.join(path, "setup.py")) as f:
            tree = ast.parse(f.read())
    except FileNotFoundError:
        return None

    # For something more complete, see:
    # https://github.com/python-poetry/poetry/blob/main/src/poetry/utils/setup_reader.py
    setup_vars = {}
    setup_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            isinstance(node.func, ast.Name)
            and node.func.id == "setup"
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "setuptools"
                and node.func.attr == "setup"
            )
        ):
            setup_call = node

        if isinstance(node, ast.Assign):
            for var in node.targets:
                if isinstance(var, ast.Name):
                    setup_vars[var.id] = node.value

    if setup_call is None:
        return None

    def entry_points_parser(x: dict[str, Any]) -> dict[str, str]:
        return dict(script.split("=", maxsplit=1) for script in x.get("console_scripts", []))

    def extras_require_parser(x: dict[str, Any]) -> dict[str, list[str]]:
        return {k: ([v] if isinstance(v, str) else v) for k, v in x.items() if v}

    setup_parser = {
        "name": ("name", lambda x: x),
        "version": ("version", lambda x: x),
        "description": ("description", lambda x: x),
        "entry_points": ("scripts", entry_points_parser),
        "install_requires": ("dependencies", lambda x: x),
        "extras_require": ("optional-dependencies", extras_require_parser),
        "python_requires": ("requires-python", lambda x: x),
        "classifiers": ("classifiers", lambda x: x),
    }

    ret = {}
    for key in setup_call.keywords:
        if key.arg in setup_parser:
            try:
                value = key.value
                if isinstance(value, ast.Name) and value.id in setup_vars:
                    value = setup_vars[value.id]
                value = ast.literal_eval(value)
            except ValueError:
                continue
            name, parser = setup_parser[key.arg]
            ret[name] = parser(value)
    return ret


class InstEditError(Exception):
    pass


class Project:
    def __init__(self, path: str) -> None:
        assert os.path.isabs(path)
        self.path = path

    @functools.cached_property
    def project(self) -> dict[str, Any]:
        try:
            with open(os.path.join(self.path, "pyproject.toml")) as f:
                pyproj = tomllib.loads(f.read())
            if "project" in pyproj and "name" in pyproj["project"]:
                return pyproj["project"]
        except FileNotFoundError:
            pass
        proj = get_project_from_setup_py(self.path)
        if proj is not None and "name" in proj:
            return proj
        raise InstEditError(f"Failed to determine project metadata in {self.path}")

    @property
    def name(self) -> str:
        return self.project["name"]

    @functools.cached_property
    def version(self) -> Version:
        # 0.0.0 comes from distutils.dist.DistributionMetadata.get_version
        return Version(self.project.get("version", "0.0.0"))

    @functools.cached_property
    def canonical_name(self) -> str:
        return canonical_name(self.name)

    @property
    def dependencies(self) -> list[str]:
        return self.project.get("dependencies", [])

    @property
    def optional_dependencies(self) -> dict[str, list[str]]:
        return self.project.get("optional-dependencies", {})

    @property
    def scripts(self) -> dict[str, str]:
        return self.project.get("scripts", {})

    @property
    def install_path(self) -> str:
        src_dir = os.path.join(self.path, "src")
        if os.path.isdir(src_dir):
            return src_dir
        return self.path


def get_prefix(python: str) -> str:
    return os.path.dirname(os.path.dirname(python))


def get_bin(python: str) -> str:
    return os.path.dirname(python)


@functools.cache
def get_purelib(python: str) -> str:
    # Approximate the logic in sysconfig
    base = get_prefix(python)

    if os.name == "nt":
        return os.path.join(base, "Lib", "site-packages")

    if python == sys.executable:
        pyname = f"python{sys.version_info[0]}.{sys.version_info[1]}"
    else:
        if re.fullmatch(r"python\d+\.\d+", os.path.basename(python)):
            pyname = os.path.basename(python)
        else:
            for p in os.scandir(get_bin(python)):
                if (
                    re.fullmatch(r"python\d+\.\d+", p.name)
                    and os.path.join(get_bin(python), os.readlink(p.path)) == python
                ):
                    pyname = p.name
                    break
            else:
                prog = 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'
                cmd = [python, "-c", prog]
                pyname = subprocess.check_output(cmd).decode().strip()

    return os.path.join(base, "lib", pyname, "site-packages")


def make_pth(proj: Project, python: str) -> list[tuple[str, int]]:
    pth = os.path.join(get_purelib(python), proj.canonical_name + ".pth")
    contents = proj.install_path + "\n"
    with open(pth, "w") as f:
        f.write(contents)
    return [(pth, len(contents))]


def make_entry_points(proj: Project, python: str) -> list[tuple[str, int]]:
    # https://setuptools.pypa.io/en/latest/userguide/entry_point.html#console-scripts
    assert len(python) < 100 and " " not in python
    shebang = f"#!{python}"

    ret = []
    for script_name, entry in proj.scripts.items():
        prefix, suffix = entry.split(":", 1)
        module = prefix
        import_name = suffix.split(".")[0]
        func = suffix
        contents = rf"""{shebang}
# -*- coding: utf-8 -*-
import re
import sys
from {module} import {import_name}
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit({func}())
"""
        script = os.path.join(get_bin(python), script_name)
        try:
            os.unlink(script)
        except FileNotFoundError:
            pass
        with open(script, "w") as f:
            f.write(contents)
        os.chmod(script, (os.stat(script).st_mode | 0o555) & 0o7777)
        ret.append((script, len(contents)))
    return ret


def make_metadata(proj: Project, metadata_dir: str) -> list[tuple[str, int]]:
    # https://packaging.python.org/en/latest/specifications/core-metadata/
    # See PEP 685, 643, PEP 566, PEP 426, PEP 345, PEP 314, PEP 241

    # Note email.policy is slow, so just write stuff out naively and hope for the best
    contents = ["Metadata-Version: 2.3", f"Name: {proj.canonical_name}", f"Version: {proj.version}"]
    if summary := proj.project.get("description", "").strip():
        contents.append(f"Summary: {summary}")
    if requires_python := proj.project.get("requires-python"):
        contents.append(f"Requires-Python: {requires_python}")
    for req_str in proj.dependencies:
        contents.append(f"Requires-Dist: {req_str}")
    for extra, deps in proj.optional_dependencies.items():
        extra = canonical_name(extra)  # PEP 685
        contents.append(f"Provides-Extra: {extra}")
        for req_str in deps:
            req = Requirement(req_str)
            marker = packaging.markers.Marker("extra == '{extra}'")
            if req.marker is None:
                req.marker = marker
            else:
                req.marker._markers = [req.marker._markers, "and", marker._markers]
            contents.append(f"Requires-Dist: [{extra}] {req}")
    contents.append("\nUNKNOWN BODY\n")

    contents_str = "\n".join(contents) + "\n"
    with open(os.path.join(metadata_dir, "METADATA"), "w") as f:
        f.write(contents_str)
    return [(os.path.join(metadata_dir, "METADATA"), len(contents_str))]


def make_entry_points_txt(proj: Project, metadata_dir: str) -> list[tuple[str, int]]:
    # https://packaging.python.org/en/latest/specifications/entry-points/
    contents = []
    if scripts := proj.scripts:
        contents.append("[console_scripts]\n")
        for key, entry in scripts.items():
            contents.append(f"{key} = {entry}\n")
        contents.append("\n")
    if gui_scripts := proj.project.get("gui-scripts"):
        contents.append("[gui_scripts]\n")
        for key, entry in gui_scripts.items():
            contents.append(f"{key} = {entry}\n")
        contents.append("\n")
    if entry_points := proj.project.get("entry-points"):
        for group, entries in entry_points.items():
            contents.append(f"[{group}]\n")
            for key, entry in entries.items():
                contents.append(f"{key} = {entry}\n")
    if contents:
        with open(os.path.join(metadata_dir, "entry_points.txt"), "w") as f:
            f.write("".join(contents))
        return [(os.path.join(metadata_dir, "entry_points.txt"), len(contents))]
    return []


def make_top_level_txt(proj: Project, metadata_dir: str) -> list[tuple[str, int]]:
    # See:
    # setuptools.command.egg_info.write_toplevel_names
    # setuptools.dist.iter_distribution_names

    # setuptools uses the first component of packages, py_modules and ext_modules
    # We don't have that information, so this is a bit of a guess.
    contents = []
    for p in os.listdir(proj.path):
        stem, ext = os.path.splitext(p)
        if ext in {".py", ".so"}:
            contents.append(stem)
    contents_str = "\n".join(contents) + "\n"
    with open(os.path.join(metadata_dir, "top_level.txt"), "w") as f:
        f.write(contents_str)
    return [(os.path.join(metadata_dir, "top_level.txt"), len(contents_str))]


def make_installer(proj: Project, metadata_dir: str) -> list[tuple[str, int]]:
    # https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-installer-file
    contents = "instedit"
    with open(os.path.join(metadata_dir, "INSTALLER"), "w") as f:
        f.write(contents)
    return [(os.path.join(metadata_dir, "INSTALLER"), len(contents))]


def make_direct_url(proj: Project, metadata_dir: str) -> list[tuple[str, int]]:
    # https://packaging.python.org/en/latest/specifications/direct-url/
    contents = '{"url":"file://%s","dir_info":{"editable":true}}' % proj.path
    with open(os.path.join(metadata_dir, "direct_url.json"), "w") as f:
        assert '"' not in proj.path and "\\" not in proj.path
        f.write(contents)
    return [(os.path.join(metadata_dir, "direct_url.json"), len(contents))]


def make_record(record: list[tuple[str, int]], metadata_dir: str) -> None:
    # https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file
    base = os.path.dirname(metadata_dir)
    with open(os.path.join(metadata_dir, "RECORD"), "w") as f:
        for path, size in record:
            path = os.path.relpath(path, base)
            assert "," not in path
            f.write(f"{path},,{size}\n")

        record_path = os.path.join(metadata_dir, "RECORD")
        record_path = os.path.relpath(record_path, base)
        f.write(f"{record_path},,\n")


def make_dist(proj: Project, python: str) -> None:
    record = []
    record += make_pth(proj, python)
    record += make_entry_points(proj, python)

    dist_info_name = f"{proj.canonical_name}-{proj.version}.dist-info"
    dist_info = os.path.join(get_purelib(python), dist_info_name)
    os.makedirs(dist_info, exist_ok=True)

    record += make_metadata(proj, metadata_dir=dist_info)
    record += make_entry_points_txt(proj, metadata_dir=dist_info)
    record += make_top_level_txt(proj, metadata_dir=dist_info)

    record += make_installer(proj, metadata_dir=dist_info)
    record += make_direct_url(proj, metadata_dir=dist_info)

    make_record(record, metadata_dir=dist_info)


def install_proj(proj: Project, python: str) -> None:
    installed = _get_purelib_installed_dists(python)
    if proj.canonical_name in installed:
        uninstall([proj.canonical_name], python)

    # Handle dependencies
    install_pypi([Requirement(req_str) for req_str in proj.dependencies], python, installed)

    print(f"Installing {proj.name}...", file=sys.stderr)
    make_dist(proj, python)


def _merge_reqs(reqs: list[Requirement]) -> Requirement:
    assert reqs
    combined = Requirement(str(reqs[0]))
    for req in reqs[1:]:
        # It would be nice if there was an officially sanctioned way of combining these
        if combined.url and req.url and combined.url != req.url:
            raise RuntimeError(f"Conflicting URLs for {combined.name}: {combined.url} vs {req.url}")
        combined.url = combined.url or req.url
        combined.extras.update(req.extras)
        combined.specifier &= req.specifier
        if combined.marker and req.marker:
            # Note that if a marker doesn't pan out, it can still contribute its version specifier
            # to the combined requirement
            combined.marker._markers = [combined.marker._markers, "or", req.marker._markers]
        else:
            # If one of markers is None, that is an unconditional install
            combined.marker = None
    return combined


class _InstalledDist:
    def __init__(
        self, canonical_name: str, version: Version, dist: importlib.metadata.Distribution
    ) -> None:
        self.name = canonical_name
        self.version = version
        self.dist = dist


def _get_purelib_installed_dists(python: str) -> dict[str, _InstalledDist]:
    installed: dict[str, _InstalledDist] = {}

    context = importlib.metadata.DistributionFinder.Context(path=[get_purelib(python)])
    for dist in importlib.metadata.distributions(context=context):
        name: str | None = None
        version: Version | None = None

        # As an optimisation, get name and version from dist-info directory name if possible,
        # instead of parsing metadata. This is technically a little unsafe (it's easier to have
        # just a stray directory than a stray directory with valid METADATA)
        if isinstance(dist, importlib.metadata.PathDistribution) and (
            m := re.fullmatch(r"(\S+)-(\S+)\.dist-info", dist._path.name)
        ):
            name = canonical_name(m.group(1))
            try:
                version = packaging.version.parse(m.group(2))
            except packaging.version.InvalidVersion:
                pass

        if name is None or version is None:
            # This likely only happens for egg-info
            metadata = dist.metadata  # this property is expensive
            dist_name = metadata["Name"]
            if dist_name is None:
                # this can happen if you have a stray empty .egg-info directory
                continue
            name = canonical_name(dist_name)
            version = packaging.version.parse(metadata["Version"])

        installed[name] = _InstalledDist(name, version, dist)

    return installed


def _filter_satisfied_requirements(
    reqs: list[Requirement], python: str, installed: dict[str, Version]
) -> list[Requirement]:

    def is_base_installed(req: Requirement) -> bool:
        if req.marker is not None:
            try:
                # If the marker is not satisfied, there is effectively no requirement
                # and we can just return True
                if not req.marker.evaluate():
                    return True
            except packaging.markers.UndefinedEnvironmentName:
                # Likely because the marker conditions install on an extra. We don't have any
                # extras in this context, so again there is effectively no requirement
                return True
        dist = installed.get(canonical_name(req.name))
        return dist is not None and req.specifier.contains(dist.version, prereleases=True)

    resolved: dict[str, list[Requirement]] = {}
    unresolved: dict[str, list[Requirement]] = {}
    for req in reqs:
        unresolved.setdefault(canonical_name(req.name), []).append(req)

    # Everything in this loop is just to resolve extras. Extras are complicated!
    while unresolved:
        name, reqs = unresolved.popitem()
        combined = _merge_reqs(reqs)
        assert name == canonical_name(combined.name)
        resolved.setdefault(name, []).append(combined)

        if is_base_installed(combined) and combined.extras:
            # We're trying to install `package[extra1, extra2]`. The right version of `package`
            # is installed, so we could potentially skip it! But we need to check whether the extra
            # requirements are installed.
            for req_str in installed[name].dist.requires or []:
                req = Requirement(req_str)
                if req is None:
                    continue
                # If req does not belong to combined.extras (e.g. package's extra1 or extra2),
                # we can ignore it
                if req.marker is None or all(
                    not req.marker.evaluate({"extra": extra}) for extra in combined.extras
                ):
                    continue
                # Get rid of the marker, since we know we want it
                req.marker = None
                req_name_canonical = canonical_name(req.name)

                # If we've already resolved this specific requirement, we can ignore it
                if req in resolved.get(req_name_canonical, []):
                    continue
                # If the extra requirement is not installed or doesn't have extras, things are
                # straightforward. We know we won't want anything other than req as a result of req,
                # so we can just add it to resolved.
                if not is_base_installed(req) or not req.extras:
                    resolved.setdefault(req_name_canonical, []).append(req)
                    continue
                # However, if the requirement is installed and has extras, we need to check whether
                # those extras are installed, essentially recursively. That is, this is the case
                # where you have a package that has:
                #     extras_require={'extra1': 'req[another_extra] == 1.0'}
                # so `req` looks something like:
                #     Requirement("req[another_extra] == 1.0 ; extra == 'extra1'")
                # So now we need to ensure that all the things implied by `req[another_extra]`
                # are installed.
                # The thing that prevents us from looping infinitely is the check above for whether
                # we've already resolved this specific requirement
                unresolved.setdefault(req_name_canonical, []).append(req)

    # The nice simple logic, now that we've resolved all extras
    to_install = []
    for reqs in resolved.values():
        combined = _merge_reqs(reqs)
        if is_base_installed(combined):
            continue
        to_install.append(combined)
    return to_install


def uninstall(dists: list[str], python: str) -> None:
    if not dists:
        return

    env = os.environ.copy()
    if shutil.which("uv"):
        env["VIRTUAL_ENV"] = get_prefix(python)
        cmd = ["uv", "pip", "uninstall", *dists]
    else:
        pip = os.path.join(get_bin(python), "pip")
        if os.path.exists(pip):
            cmd = [python, "-m", "pip"]
        else:
            cmd = [sys.executable, "-m", "pip", "--python", python]
        cmd += ["--disable-pip-version-check", "--progress-bar=off"]
        cmd += ["uninstall", "--yes", *dists]

    print("Running:", shlex.join(cmd), file=sys.stderr)
    subprocess.run(cmd, env=env, check=True)


def install_pypi(pypi_deps: list[Requirement], python: str, installed: dict[str, Version] | None) -> None:
    if not pypi_deps:
        return

    to_install = _filter_satisfied_requirements(pypi_deps, python, installed)
    if not to_install:
        return

    env = os.environ.copy()
    if shutil.which("uv"):
        # If we have uv maybe we don't need to bother with _filter_satisfied_requirements
        env["VIRTUAL_ENV"] = get_prefix(python)
        cmd = ["uv", "pip", "install"]
    else:
        pip = os.path.join(get_bin(python), "pip")
        if os.path.exists(pip):
            cmd = [python, "-m", "pip"]
        else:
            cmd = [sys.executable, "-m", "pip", "--python", python]
        cmd += ["--disable-pip-version-check", "--progress-bar=off"]
        cmd += ["install"]
    for req in to_install:
        cmd.append(str(req))

    print("Running:", shlex.join(cmd), file=sys.stderr)
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default=".", help="Path to the project to install editably"
    )
    parser.add_argument(
        "--python",
        help="Python to install to (defaults to Python from VIRTUAL_ENV if set, otherwise sys.executable)",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    python = args.python
    if not python:
        if "VIRTUAL_ENV" in os.environ:
            prefix = os.environ["VIRTUAL_ENV"]
            if os.name == "nt":
                python = os.path.join(prefix, "Scripts", "python.exe")
            else:
                python = os.path.join(prefix, "bin", "python")
        else:
            python = sys.executable
    python = os.path.abspath(python)

    try:
        install_proj(Project(path), python)
    except InstEditError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

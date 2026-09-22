%global debug_package %{nil}

# Distros that ship pyproject-rpm-macros (Fedora, EL9+) build via the
# %%pyproject_* macro family -- that path is unchanged. Distros without it
# (openEuler 24.03, EL8-family) fall back to a plain pip wheel/install build.
# Capability-detected at parse time, so the build container must have its
# python toolchain installed before rpmbuild runs.
%if %{defined pyproject_wheel}
%global has_pyproject_macros 1
%else
%global has_pyproject_macros 0
%endif

# Filter the auto-generated Requires for: torch.
# Reason: distro torch is CPU-only; users install GPU torch via pip.
# See packaging/INSTALL.md (or future flagos-packaging install docs) for the
# user-side pip install incantation.
%global __requires_exclude ^python3(\.[0-9]+)?dist\((torch)\)$
Name:           python3-flag-blas
Version:        0.3.0
Release:        1%{?dist}
Summary:        FlagBLAS — linear-algebra kernels for FlagOS

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagBLAS
Source0:        %{url}/archive/refs/tags/v%{version}.tar.gz#/flag-blas-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
%if %{has_pyproject_macros}
BuildRequires:  pyproject-rpm-macros
%endif

%description
GEMM and related BLAS-level operators implemented in Triton, targeting FlagOS multi-vendor accelerators.

%prep
%autosetup -n flag-blas-%{version}

%build
%if %{has_pyproject_macros}
%pyproject_wheel
%else
%{__python3} -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
%endif

%install
%if %{has_pyproject_macros}
%pyproject_install
%pyproject_save_files flag_blas
%else
%{__python3} -m pip install --no-deps --no-index --no-warn-script-location \
    --root %{buildroot} --prefix /usr dist/*.whl
%endif

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flag_blas'); assert s and s.origin, 'flag_blas not findable'; print('OK: flag_blas at', s.origin)"

%if %{has_pyproject_macros}
%files -f %{pyproject_files}
%else
%files
%{python3_sitelib}/flag_blas/
%{python3_sitelib}/flag_blas-%{version}.dist-info/
%endif
%license LICENSE

%changelog
* Thu Sep 17 2026 FlagOS Contributors <contact@flagos.io> - 0.3.0-1
- Align the packaging baseline with the 0.3.0 release line.

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.

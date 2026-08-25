# Third-party dependencies

Clone the pinned dependencies after checking out the parent repository:

```bash
git submodule update --init --recursive
```

OmniDrones is consumed as an editable package so changes inside its submodule
are immediately visible to the tutorial environments. Install it in the active
Isaac Lab Python environment without re-resolving its legacy dependency pins:

```bash
python -m pip install --no-deps -e source/thirdparty/OmniDrones
```

Verify that Python resolves the project-local checkout:

```bash
python -c "import omni_drones; print(omni_drones.__file__)"
```

The reported path should be under `source/thirdparty/OmniDrones`.

python_requirements(
    name="reqs0",
)

python_sources(
    name="root",
)


pex_binary(
    name="babbar.blip.pex",
    entry_point="annotate.py",
    dependencies=["//:reqs0"],
    shebang="/usr/bin/env python3",
    output_path="babbar.blip.annotate.pex",
    include_tools=True,
)
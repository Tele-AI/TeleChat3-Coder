from setuptools import setup, find_packages

setup(
    name="telechat3_vllm_patch",
    version="1.1.0",
    packages=find_packages(
        include=["telechat3_tool_parser.py", "telechat3_reasoning.py"]
    ),
    entry_points={
        "vllm.general_plugins": [
            "telechat3_tool_parser = telechat3_tool_parser:register_tool_parser",
            "telechat3_reasoning = telechat3_reasoning:register_reasoning",
        ]
    },
)

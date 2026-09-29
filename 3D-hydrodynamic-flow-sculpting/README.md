# 3D-hydrodynamic-flow-sculpting

A tool for designing flow-sculpting microfluidic channels and simulating resulting flow profiles.

## Precomputed Libraries

You can download the precomputed libraries here:
https://drive.google.com/drive/folders/16BtjLV74uk7A9ZAgGJoo5kOc5aUSpx3p?usp=drive_link

after downloading, place them in the corresponding folders:

```text
project_root/
├── generated_library_inlet/
├── generated_library_outlet/
├── generated_library_pillar/
├── visualization_v2.py
```

## Quick Start

run the visualization tool:

```bash
python visualization_v2.py
```

## Generate Your Own Libraries

you can also generate your own libraries with raw COMSOL streamline data.

Prepare streamline data exported from COMSOL and place them in the following folders:

```text
streamline_files_inlet/
streamline_files_outlet/
streamline_files_pillar/
```

Run the following scripts from the project root directory:

```bash
python library_generation_inlet.py
python library_generation_outlet.py
python library_generation_pillar.py
```

The generated libraries will be saved to:

```text
generated_library_inlet/
generated_library_outlet/
generated_library_pillar/
```

These can then be used directly by the main program.


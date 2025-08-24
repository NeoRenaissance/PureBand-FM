# PureBand-FM

An open-source in-band digital FM standard project to embed digital data within the 200 kHz FM envelope, reducing interference and enabling more broadcasters. Includes simulations, data collection tools, and GRC flowgraphs.

## Folders
- `simulations/`: Python scripts for SNR, BER, and antenna comparisons.
- `data_collection/`: Heat map data collection with web GUI.
- `docs/`: Pitch decks, flowcharts, and comparisons.
- `grc_files/`: GNU Radio Companion flowgraphs for transmitter/receiver.
- `wav_generation/`: Scripts to generate theoretical .wav files for SDR++.

## Setup
- Install dependencies: `brew install gnuradio librtlsdr` (macOS).
- Python libs: `pip install numpy scipy pyrtlsdr pynmea2 flask pyobjc galois`.
- Run simulations: `python simulations/fm_simulation.py`.

## License
MIT

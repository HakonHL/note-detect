Axon Audio
##########

Audio processing module for note detection using FFT and chroma analysis.

Overview
********

This module processes audio input and extracts chroma features using FFT-based analysis. The implementation runs on the Axon NPU and supports both synchronous and asynchronous inference modes.

Configuration
*************

Configuration options are available in ``prj.conf`` and ``Kconfig``.

Files
*****

- ``src/main.c`` - Main application entry point
- ``src/fft_chroma.c/h`` - FFT and chroma analysis implementation
- ``CMakeLists.txt`` - Build configuration
- ``sample.yaml`` - Sample metadata

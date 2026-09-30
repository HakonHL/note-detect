#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(hello_axon)

# Source files
target_sources(app PRIVATE
    src/main.c
    src/fft_chroma.c
    ../../../applications/ww_kws/src/i2s_audio.c
)
target_include_directories(app PRIVATE ../../../applications/ww_kws/src)
target_compile_definitions(app PRIVATE CONFIG_APP_AUDIO_I2S_CHANNEL=0)

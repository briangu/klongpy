cmake_minimum_required(VERSION 3.16)
# Only the components the console needs (and what they require): a smaller, faster build.
set(COMPONENTS main)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(klong_esp32p4)

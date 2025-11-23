cmake_minimum_required(VERSION 3.15...3.27)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

include_directories(
    "c++/include"
    "c++/vendor/eigen"
    "python/celerite2")

pybind11_add_module(driver "python/celerite2/driver.cpp")
target_compile_features(driver PUBLIC cxx_std_14)
install(TARGETS driver LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

pybind11_add_module(backprop "python/celerite2/backprop.cpp")
target_compile_features(backprop PUBLIC cxx_std_14)
install(TARGETS backprop LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

option(BUILD_JAX "Build JAX extension (requires jaxlib headers)" ON)
if(BUILD_JAX)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import sys, pathlib; import jaxlib; path = pathlib.Path(jaxlib.__file__).parent / 'include'; sys.stdout.write(str(path) if path.exists() else '')"
        OUTPUT_VARIABLE JAX_INCLUDE
        RESULT_VARIABLE JAXLIB_RES
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(JAXLIB_RES EQUAL 0 AND NOT "${JAX_INCLUDE}" STREQUAL "")
        message(STATUS "Building JAX extension with jaxlib include: ${JAX_INCLUDE}")
        pybind11_add_module(xla_ops "python/celerite2/jax/xla_ops.cpp")
        target_compile_features(xla_ops PUBLIC cxx_std_17)
        target_include_directories(xla_ops PRIVATE "${JAX_INCLUDE}")
        install(TARGETS xla_ops LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/jax")
    else()
        message(STATUS "Skipping JAX extension (jaxlib not found)")
    endif()
endif()

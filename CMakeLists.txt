cmake_minimum_required(VERSION 2.8)

project(bayrepo_neuro_farm)

execute_process (
    COMMAND bash -c "make -C civetweb COPT='-DNO_SSL -DNO_SSL_DL -g -O0' lib"
)

add_compile_options(-g -O0)

set(SOURCE_EXE neuro_web.c neuro.c)

set(SOURCE_LIB neuro_web_client.c)

add_executable(bayrepo_neuro_farm ${SOURCE_EXE})

add_library(bayrepo_nfl STATIC ${SOURCE_LIB})

target_link_libraries(bayrepo_neuro_farm pthread gsl gslcblas m png crypto ${CMAKE_SOURCE_DIR}/civetweb/libcivetweb.a)
target_link_libraries(bayrepo_nfl curl)
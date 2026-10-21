add_executable(scratch_bin main.cpp
        common.h
        timing.h
        checks.h
        estimator.h)
target_link_libraries(scratch_bin PRIVATE polyfem::polyfem)
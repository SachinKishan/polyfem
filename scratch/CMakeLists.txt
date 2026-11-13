add_executable(scratch_bin main.cpp
        common.h
        timing.h
        checks.h
        estimator.h
        check_estimators.h
        command_line.h
        resume.h
        run_adaptive.h
        run_single.h
        run_summary.h
        run_uniform.h
        stopping.h)
target_link_libraries(scratch_bin PRIVATE polyfem::polyfem)
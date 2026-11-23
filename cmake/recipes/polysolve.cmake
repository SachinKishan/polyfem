# PolySolve (https://github.com/polyfem/polysolve)
# License: MIT

if(TARGET polysolve)
    return()
endif()

message(STATUS "Third-party: creating target 'polysolve'")

include(CPM)

CPMDeclarePackage(eigen NAME eigen GIT_REPOSITORY https://eigen.googlesource.com/mirror GIT_TAG 5.0.1 DOWNLOAD_ONLY ON)
# polysolve.patch: EigenDirect throws on any factorize/solve failure (Pardiso errors other than a zero
# pivot used to pass silently and return garbage), and adds Eigen::PardisoLDLT64 (Pardiso with 64-bit
# indices through MKL's pardiso_64, for systems whose factor has more than 2^31 nonzeros)
CPMAddPackage(
    NAME polysolve
    GITHUB_REPOSITORY polyfem/polysolve
    GIT_TAG 621e8fb3889717a33c4fa33a608052e9d7459e63
    PATCHES "${CMAKE_CURRENT_LIST_DIR}/polysolve.patch"
)

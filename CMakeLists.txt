##
#  CMake script for efsim
##

# Set the name of the project and target:
SET(TARGET "efsim")

# Declare all source files the target consists of:
SET(TARGET_SRC
  programs/efsim.cc
  # You can specify additional files here!
  )


CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

ADD_EXECUTABLE(spsim programs/spsim.cc)
DEAL_II_SETUP_TARGET(spsim)

ADD_EXECUTABLE(run_convergence_test programs/run_convergence_test.cc)
DEAL_II_SETUP_TARGET(run_convergence_test)

ADD_EXECUTABLE(run_convergence_test_parabolic programs/run_convergence_test_parabolic.cc)
DEAL_II_SETUP_TARGET(run_convergence_test_parabolic)

ADD_EXECUTABLE(test_postprocessMM tests/test_postprocessMM.cc)
DEAL_II_SETUP_TARGET(test_postprocessMM)

ADD_EXECUTABLE(test_transport tests/test_transport.cc)
DEAL_II_SETUP_TARGET(test_transport)

ADD_EXECUTABLE(test_elliptic_solver tests/test_elliptic_solver.cc)
DEAL_II_SETUP_TARGET(test_elliptic_solver)

ADD_EXECUTABLE(test_residual tests/test_residual.cc)
DEAL_II_SETUP_TARGET(test_residual)

ADD_EXECUTABLE(test_parabolic_solver tests/test_parabolic_solver.cc)
DEAL_II_SETUP_TARGET(test_parabolic_solver)

ADD_EXECUTABLE(test_dirichlet_flux_recovery tests/test_dirichlet_flux_recovery.cc)
DEAL_II_SETUP_TARGET(test_dirichlet_flux_recovery)

ADD_EXECUTABLE(test_rock_prop tests/test_rock_prop.cc)
DEAL_II_SETUP_TARGET(test_rock_prop)

ADD_EXECUTABLE(test_vtk_reader tests/test_vtk_reader.cc)
DEAL_II_SETUP_TARGET(test_vtk_reader)

ADD_EXECUTABLE(test_embedded_surface tests/test_embedded_surface.cc)
DEAL_II_SETUP_TARGET(test_embedded_surface)

file(COPY input_files/8cellmodel.vtk DESTINATION ${CMAKE_BINARY_DIR})
file(COPY input_files/grid_four_cells_distorted.ucd DESTINATION ${CMAKE_BINARY_DIR})
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/output)

enable_testing()
add_test(TransportSolver test_transport)
add_test(EllipticSolver test_elliptic_solver)
add_test(PostProcessMM test_postprocessMM)
add_test(Residual test_residual)
add_test(ParabolicSolver test_parabolic_solver)
add_test(DirichletFluxRecovery test_dirichlet_flux_recovery)
add_test(TestRockProperties test_rock_prop)
add_test(VTKReader test_vtk_reader 8cellmodel.vtk)
add_test(EmbeddedSurface test_embedded_surface)

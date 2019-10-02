cmake_minimum_required (VERSION 3.0.2)
project (DecisionServingPlatform VERSION 0.0.1)

add_subdirectory(external/googletest)
enable_testing()

option(CODE_COVERAGE "Enable code coverage" OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDSP_OSS -DDSP_VERSION='\"${PROJECT_VERSION}\"' -Wno-deprecated-declarations")

IF(CODE_COVERAGE)
  if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif()
ENDIF(CODE_COVERAGE)

# Enable C++-17
set(CMAKE_CXX_STANDARD 17)

# Enable debug info
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

# Add cmake script directory.
LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Required packages
find_package(Threads REQUIRED)
find_package(gflags REQUIRED)
find_package(Folly REQUIRED)
find_package(glog CONFIG)
find_package(wangle CONFIG)
find_package(Boost REQUIRED COMPONENTS system thread filesystem regex context program_options)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  external
  external/exprtk

  .

  ${TORCH_INCLUDE_DIRS}
  ${FOLLY_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/gen/gen-cpp2
  ${PROJECT_BINARY_DIR}/gen
)

add_library(
  dsp-lib
  STATIC

  ${api-cpp2-HEADERS}
  ${api-cpp2-SOURCES}
  ${decision-cpp2-HEADERS}
  ${decision-cpp2-SOURCES}
  ${example-cpp2-HEADERS}
  ${example-cpp2-SOURCES}

  reagent/serving/core/DecisionServiceException.cpp
  reagent/serving/core/ConfigProvider.cpp
  reagent/serving/core/ActionValueScorer.cpp
  reagent/serving/core/PytorchActionValueScorer.cpp
  reagent/serving/core/Operator.cpp
  reagent/serving/core/DecisionPlan.cpp
  reagent/serving/core/OperatorRunner.cpp
  reagent/serving/core/SharedParameterHandler.cpp
  reagent/serving/core/OperatorFactory.cpp
  reagent/serving/core/LogJoiner.cpp
  reagent/serving/core/InMemoryLogJoiner.cpp
  reagent/serving/core/RealTimeCounter.cpp
  reagent/serving/core/Headers.cpp
  reagent/serving/core/LocalRealTimeCounter.cpp
  reagent/serving/core/DecisionService.cpp

  reagent/serving/operators/Ucb.cpp
  reagent/serving/operators/ActionValueScoring.cpp
  reagent/serving/operators/PropensityFit.cpp
  reagent/serving/operators/Softmax.cpp
  reagent/serving/operators/SoftmaxRanker.cpp
  reagent/serving/operators/Frechet.cpp
  reagent/serving/operators/Expression.cpp
)

add_executable(
  DspCli

  reagent/serving/cli/Main.cpp
)
target_link_libraries (
  DspCli
  LINK_PUBLIC

  dsp-lib
  ${TORCH_LIBRARIES}
  Folly::folly
  glog::glog
  ${GFLAGS_LIBRARIES}
  gtest
  gtest_main
  gmock
  gmock_main
)

add_executable(
  DspTest

  reagent/serving/test/DecisionService_test.cpp
  reagent/serving/test/InMemoryLogJoiner_test.cpp
  reagent/serving/test/Frechet_test.cpp
  reagent/serving/test/Ucb_test.cpp
  reagent/serving/test/PlanProvider_test.cpp
  reagent/serving/test/PropensityFit_test.cpp
  reagent/serving/test/Softmax_test.cpp
  reagent/serving/test/SoftmaxRanker_test.cpp
  reagent/serving/test/TestHeaders.cpp
  reagent/serving/test/Expression_test.cpp
  reagent/serving/test/PytorchScoring_test.cpp
)
target_link_libraries (
  DspTest
  LINK_PUBLIC

  dsp-lib
  ${TORCH_LIBRARIES}
  Folly::folly
  glog::glog
  ${GFLAGS_LIBRARIES}
  gtest
  gtest_main
  gmock
  gmock_main
)
add_test(
  DspTest
  DspTest
)


install(TARGETS DspCli
  PERMISSIONS  OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
  DESTINATION "bin"
  )

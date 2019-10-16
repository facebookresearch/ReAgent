cmake_minimum_required (VERSION 3.0.2)
project (ReAgentServingPlatform VERSION 0.0.1)

add_subdirectory(external/googletest)
enable_testing()

option(CODE_COVERAGE "Enable code coverage" OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRASP_OSS -DRASP_VERSION='\"${PROJECT_VERSION}\"' -Wno-deprecated-declarations")

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
find_package(glog CONFIG)
find_package(Boost REQUIRED COMPONENTS system thread filesystem regex context program_options)
find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(
  external
  external/exprtk
  external/cpp-taskflow
  external/nlohmann_json/single_include

  .

  ${TORCH_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/gen/gen-cpp2
  ${PROJECT_BINARY_DIR}/gen
)

add_library(
  rasp-lib
  STATIC

  ${api-cpp2-HEADERS}
  ${api-cpp2-SOURCES}
  ${decision-cpp2-HEADERS}
  ${decision-cpp2-SOURCES}
  ${example-cpp2-HEADERS}
  ${example-cpp2-SOURCES}

  reagent/serving/core/DecisionServiceException.cpp
  reagent/serving/core/ConfigProvider.cpp
  reagent/serving/core/DiskConfigProvider.cpp
  reagent/serving/core/Containers.cpp
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
  reagent/serving/operators/EpsilonGreedyRanker.cpp
  reagent/serving/operators/PropensityFit.cpp
  reagent/serving/operators/Softmax.cpp
  reagent/serving/operators/SoftmaxRanker.cpp
  reagent/serving/operators/Frechet.cpp
  reagent/serving/operators/Expression.cpp
)

add_executable(
  RaspCli

  reagent/serving/cli/Main.cpp

  reagent/serving/cli/Server.h
  reagent/serving/cli/Server.cpp
)
target_link_libraries (
  RaspCli
  LINK_PUBLIC

  rasp-lib
  ${TORCH_LIBRARIES}
  glog::glog
  ${GFLAGS_LIBRARIES}
  Boost::filesystem
)
if (WIN32)
    set_target_properties(RaspCli PROPERTIES
        LINK_FLAGS "/WHOLEARCHIVE"
    )
elseif (APPLE)
    set_target_properties(RaspCli PROPERTIES
        LINK_FLAGS "-Wl,-all_load"
    )
else ()
    set_target_properties(RaspCli PROPERTIES
        LINK_FLAGS "-Wl,--whole-archive"
    )
endif ()

add_executable(
  RaspTest

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
  RaspTest
  LINK_PUBLIC

  rasp-lib
  ${TORCH_LIBRARIES}
  glog::glog
  ${GFLAGS_LIBRARIES}
  gtest
  gtest_main
  Boost::filesystem
)
if (WIN32)
    set_target_properties(RaspTest PROPERTIES
        LINK_FLAGS "/WHOLEARCHIVE"
    )
elseif (APPLE)
    set_target_properties(RaspTest PROPERTIES
        LINK_FLAGS "-Wl,-all_load"
    )
else ()
    set_target_properties(RaspTest PROPERTIES
        LINK_FLAGS "-Wl,--whole-archive"
    )
endif ()
add_test(
  RaspTest
  RaspTest
)


install(TARGETS RaspCli
  PERMISSIONS  OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
  DESTINATION "bin"
  )

cmake_minimum_required(VERSION 3.30)
project(NetworksFromScratch)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(extern/eigen)
include_directories(extern/eigen)

add_subdirectory(extern/EigenRand)
include_directories(extern/EigenRand)


file(COPY ${CMAKE_SOURCE_DIR}/mnist_train.csv ${CMAKE_SOURCE_DIR}/mnist_test.csv
        DESTINATION ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(NetworksFromScratch main.cpp Layer.cpp
        ActivationFunction.cpp
        LossFunction.cpp
        Network.cpp
        DataLoader.cpp
        ReadCSV.cpp
        FileReader.cpp
        NetworkBuilder.cpp
        FileWriter.cpp
        AdamTrainer.cpp
        Random.cpp
        BaseTrainer.cpp
        SGDTrainer.cpp
        GDTrainer.cpp
        SGDWithMomentumTrainer.cpp
)


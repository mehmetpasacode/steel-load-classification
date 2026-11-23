cmake_minimum_required(VERSION 3.15)
project(steel_knn LANGUAGES CXX)
add_executable(steel_knn knn.cpp)
target_compile_features(steel_knn PRIVATE cxx_std_17)

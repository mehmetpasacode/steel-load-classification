#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <queue>
#include <utility>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <sstream>
#include <limits>

std::pair<float, int> distance(int dim, int dim_of_output, float* target, std::ifstream& neighbor_file) {
   float result = 0.0;
   float neighbor;
   for(int col = 0; col < dim - dim_of_output; col++) {
       neighbor_file >> neighbor;
       result += (target[col] - neighbor) * (target[col] - neighbor);
   }
   result = std::sqrt(result);
   int result_vote = 0;
   for(int vote = 0; vote < dim_of_output; vote++) {
       neighbor_file >> neighbor;
       if(neighbor == 1) {
           result_vote = vote;
       }
   }
   return {result, result_vote};
}


int knn(int num_of_features, int num_of_neighbors, int dimension_of_output, int input_index, int
       num_of_samples, std::string data_file_name, bool do_print) {
   float* target = new float[num_of_features];
   std::priority_queue<std::pair<float, int>> top_k_neighbors;
   std::ifstream dataFile(data_file_name);
   if (!dataFile.is_open()) {
       std::cout << "Error: Could not open " << data_file_name << "\n";
       return 1;
   }
   std::string line;
   for(int i = 0; i <= input_index; i++) {
       std::getline(dataFile, line);
   }
   std::stringstream stream_row(line);
   for(int n = 0; n < num_of_features; n++) {
       stream_row >> target[n];
   }
   dataFile.clear();
   dataFile.seekg(0, std::ios::beg);
   for(int row = 0; row < num_of_neighbors; row++) {
       std::pair<float, int> result = distance(num_of_features, dimension_of_output, target, dataFile);
       if(result.first == 0.0) {
           top_k_neighbors.push({std::numeric_limits<float>::max(), 0});
       }
       else{
           top_k_neighbors.push(result);
       }
   }
   for(int row = num_of_neighbors; row < num_of_samples; row++) {
       std::pair<float, int> cur_result = distance(num_of_features, dimension_of_output, target, dataFile);
       if ((cur_result.first < top_k_neighbors.top().first) && (cur_result.first != 0.0)) {
           top_k_neighbors.pop();
           top_k_neighbors.push(cur_result);
       }
   }
   int* vote_counts = new int[dimension_of_output]();
   for(int neighbor = 0; neighbor < num_of_neighbors; neighbor++) {
       for(int dim = 0; dim < dimension_of_output; dim++) {
           if(top_k_neighbors.top().second == dim) {
               vote_counts[dim]++;
           }
       }
       top_k_neighbors.pop();
   }
   int final_result = 0;
   int result_vote_cnt = 0;
   for(int dim = 0; dim < dimension_of_output; dim++) {
       if(vote_counts[dim] > result_vote_cnt) {
           final_result = dim;
           result_vote_cnt = vote_counts[dim];
       }
   }
   if(do_print) {
       for(int i = 0; i < dimension_of_output; i++) {
           if(target[num_of_features - dimension_of_output + i] == 1) {
               final_result = (final_result == i) ? 1 : 0;
               //std::cout << i << ",";
           }
       }
   }
   delete[] target;
   delete[] vote_counts;
   return final_result;
}

int main(int argc, char* argv[]) {
   // We expect: [program_name] [info_file] [data_file] [test group percentage] [k(number of neighbors)]
   if (argc != 5) {
       std::cerr << "Error: Invalid number of arguments." << std::endl;
       return 1;
   }

   std::string info_file_name = argv[1];
   std::string data_file_name = argv[2];

   std::ifstream infoFile(info_file_name);
   //std::ifstream dataFile(data_file_name);

   if (!infoFile.is_open()) {
       std::cout << "Error: Could not open " << info_file_name << "\n";
       return 1;
   }


   int test_percentage_int = std::stoi(argv[3]);
   if((test_percentage_int <= 0) || (test_percentage_int >= 100)) {
       std::cerr << "Error: Invalid test group precentage." << std::endl;
       return 1;
   }

   int num_of_neighbors = std::stoi(argv[4]);
   if(num_of_neighbors <= 0) {
       std::cerr << "Error: Invalid number of neighbors" << std::endl;
       return 1;
   }

   int sample_size;
   int num_of_features;
   infoFile >> sample_size >> num_of_features;

   float test_percentage = (float) test_percentage_int / 100;
   int num_of_tests = sample_size * test_percentage;
   int size_of_all_fetaures = 0;
   int prev = sample_size;
   int current_type;
   while (infoFile >> current_type) {
       size_of_all_fetaures += current_type;
       if(current_type == 2) {
           size_of_all_fetaures--;
       }
       prev = current_type;
   }
   int dimension_of_output = prev;
   //Generating random test group by given percentage
   std::vector<int> row_indices(sample_size);
   std::iota(row_indices.begin(), row_indices.end(), 0);
   std::random_device rd;
   std::mt19937 g(rd());
   std::shuffle(row_indices.begin(), row_indices.end(), g);
   for (int i = 0; i < num_of_tests; i++) {
       for(int n = 3; n < 100; n += 8) {
           int result = knn(size_of_all_fetaures, n, dimension_of_output, row_indices[i], sample_size,
       data_file_name, 1);
           std::cout << result;
           if(n < 96) {
               std::cout << ",";
           }
       }
       std::cout << std::endl;

   }

   return 0;
}

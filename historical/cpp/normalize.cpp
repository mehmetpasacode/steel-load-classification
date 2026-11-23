#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <vector>
#include <algorithm>
#include <iterator>


#define ONE_HOT_VALUE 1


int main(int argc, char* argv[]) {
   // We expect: [program_name] [data_file] [info_file]
   if (argc != 3) {
       std::cout << "Error: Invalid number of arguments.\n";
       return 1;
   }

   std::string data_file_name = argv[1];
   std::string info_file_name = argv[2];

   std::ifstream infoFile(info_file_name);

   if (!infoFile.is_open()) {
       std::cout << "Error: Could not open " << info_file_name << "\n";
       return 1;
   }

   int sample_size = 0;
   int num_features = 0;
   int features_vector_size = 0;
   std::vector<std::vector<std::string>> distinct_categories;

   infoFile >> sample_size >> num_features;

   distinct_categories.resize(num_features);

   std::vector<int> feature_types(num_features);
   int current_type;

   int info_index = 0;
   while (infoFile >> current_type) {
       std::cout << "Read: " << current_type << std::endl;
       if(current_type > 2) {
           features_vector_size += current_type;
           distinct_categories[info_index].reserve(current_type);
       }
       else if(current_type == 2) {
           features_vector_size++;
           distinct_categories[info_index].reserve(current_type);
       }
       else {
           features_vector_size++;
       }
       feature_types[info_index] = current_type;
       info_index++;
   }
   std::cout << "Feature vector size: " << features_vector_size << std::endl;
   infoFile.close();

   std::string line;
   std::string cell;
   float cell_value;

   std::ifstream dataFile(data_file_name);

   if (!dataFile.is_open()) {
       std::cout << "Error: Could not open " << data_file_name << "\n";
       return 1;
   }

   std::vector<float> max_values(num_features, std::numeric_limits<float>::lowest());
   std::vector<float> min_values(num_features, std::numeric_limits<float>::max());
   bool is_first_row = true;
   float* data_normalized = new float[sample_size * features_vector_size];
   int line_index = 0;
   //Read one row of CSV
   while (std::getline(dataFile, line)) {
       if (is_first_row) {
           if (line.size() >= 3 && line.substr(0, 3) == "\xEF\xBB\xBF") {
               line = line.substr(3);
           }
           is_first_row = false;
       }
       std::stringstream ss(line);

       int inline_index = 0;
       int vector_index = 0;

       //Read comma separated values in loop
       while (std::getline(ss, cell, ',')) {
           if(feature_types[inline_index] < 2) {
               try {
                   cell_value = std::stof(cell);
               } catch (const std::invalid_argument& e) {
                   std::cout << "Invalid data: " << cell << "\n";
                   return 1;
               }
               if(cell_value > max_values[inline_index]) {
                   max_values[inline_index] = cell_value;
               }
               else if(cell_value < min_values[inline_index]) {
                   min_values[inline_index] = cell_value;
               }
               data_normalized[line_index * features_vector_size + vector_index] = cell_value;
               vector_index++;
           }
           else {
               std::vector<std::string>& currentColumnCategories = distinct_categories[inline_index];
               auto it = std::find(currentColumnCategories.begin(), currentColumnCategories.end(), cell);

               int category_index;
               if (it == currentColumnCategories.end()) {
                   currentColumnCategories.push_back(cell);
                   category_index = currentColumnCategories.size() - 1;

               } else {
                   category_index = std::distance(currentColumnCategories.begin(), it);
               }
               if (feature_types[inline_index] == 2) {
                   data_normalized[line_index * features_vector_size + vector_index] = category_index *
       ONE_HOT_VALUE;
                   vector_index++;
               }
               else{
                   data_normalized[line_index * features_vector_size + vector_index + category_index] =
       ONE_HOT_VALUE;
                   vector_index += feature_types[inline_index];
               }

           }
           inline_index++;
       }
       line_index++;
   }

   std::cout << "\n--- Continuous Features: Min and Max ---" << std::endl;

   for (int i = 0; i < num_features; i++) {
       if (feature_types[i] < 2) {
           std::cout << "Column " << i << ": "
                   << "Min = " << min_values[i]
                   << " | Max = " << max_values[i] << std::endl;
       }
       else {
           std::cout << "Column " << i << " Categories: ";
           for (size_t j = 0; j < distinct_categories[i].size(); j++) {
               std::cout << "[" << distinct_categories[i][j] << "] ";
           }
           std::cout << "(Total: " << distinct_categories[i].size() << ")" << std::endl;
       }
   }

   //NORMALIZE

   std::cout << "NORMALIZING: " << std::endl;
   std::ofstream output_file;
   output_file.open("Preprocessed_data.txt");
   if (!output_file.is_open()) {
       std::cout << "Could not open the file." << std::endl;
       return 1;
   }
   std::cout << "Feature vector size: " << features_vector_size << std::endl;
   for(int sample_index = 0; sample_index < sample_size; sample_index++) {
       int vector_index = 0;
       int feature_index = 0;
       while(feature_index < num_features) {
           if(feature_types[feature_index] < 2) {
               //std::cout << "in if 1" << std::endl;
               data_normalized[sample_index * features_vector_size + vector_index] =
       (data_normalized[sample_index * features_vector_size + vector_index] - min_values[feature_index]) /
       (max_values[feature_index] - min_values[feature_index]);
               //std::cout << "in if 2" << std::endl;
               vector_index++;
           }
           else {vector_index += feature_types[feature_index];}
           feature_index++;
       }
       //std::cout << "Sample: " << sample_index << ": ";
       for(int i = 0; i < features_vector_size; i++) {
           output_file << data_normalized[sample_index * features_vector_size + i];
           if(i != features_vector_size - 1) {
               output_file << "\t";
           }
       }
       output_file << std::endl;
       feature_index++;
   }
   output_file.close();
   return 0;
}

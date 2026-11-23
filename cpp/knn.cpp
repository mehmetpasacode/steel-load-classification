#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

using Row = std::vector<double>;
using Neighbor = std::pair<double, std::size_t>;

double squared_distance(const Row& query, const Row& candidate) {
    double sum = 0.0;
    for (std::size_t column = 0; column < query.size(); ++column) {
        const double difference = query[column] - candidate[column];
        sum += difference * difference;
    }
    return sum;
}

// Adapted from the appendix's bounded max-heap search. Only training rows
// enter the heap. Row order breaks distance ties, including exact matches.
std::vector<std::size_t> nearest(const Row& query, const std::vector<Row>& train,
                                 std::size_t max_k) {
    std::priority_queue<Neighbor> heap;
    for (std::size_t index = 0; index < train.size(); ++index) {
        const Neighbor neighbor{squared_distance(query, train[index]), index};
        if (heap.size() < max_k) {
            heap.push(neighbor);
        } else if (neighbor < heap.top()) {
            heap.pop();
            heap.push(neighbor);
        }
    }

    std::vector<std::size_t> ordered(max_k);
    for (auto destination = ordered.rbegin(); destination != ordered.rend(); ++destination) {
        *destination = heap.top().second;
        heap.pop();
    }
    return ordered;
}

Row read_row(std::size_t columns) {
    Row row(columns);
    for (auto& value : row) {
        if (!(std::cin >> value) || !std::isfinite(value)) {
            throw std::runtime_error("Expected a finite feature value");
        }
    }
    return row;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    try {
        int train_count, query_count, columns, class_count, k_count;
        if (!(std::cin >> train_count >> query_count >> columns >> class_count >> k_count)
            || train_count <= 0 || query_count <= 0 || columns <= 0 || class_count <= 0
            || k_count <= 0) {
            throw std::runtime_error("Invalid input dimensions");
        }

        std::vector<int> neighbors(k_count);
        for (auto& k : neighbors) {
            if (!(std::cin >> k) || k < 1 || k > train_count) {
                throw std::runtime_error("Invalid k");
            }
        }
        std::vector<int> labels(train_count);
        for (auto& label : labels) {
            if (!(std::cin >> label) || label < 0 || label >= class_count) {
                throw std::runtime_error("Invalid training class");
            }
        }
        std::vector<Row> train;
        train.reserve(train_count);
        for (int row = 0; row < train_count; ++row) {
            train.push_back(read_row(columns));
        }

        const int max_k = *std::max_element(neighbors.begin(), neighbors.end());
        for (int row = 0; row < query_count; ++row) {
            const auto ordered = nearest(read_row(columns), train, max_k);
            std::vector<int> votes(class_count, 0);
            std::vector<int> winners(max_k);
            for (int index = 0; index < max_k; ++index) {
                ++votes[labels[ordered[index]]];
                // First maximum wins, in the label order declared by Python.
                winners[index] = static_cast<int>(
                    std::max_element(votes.begin(), votes.end()) - votes.begin()
                );
            }
            for (int index = 0; index < k_count; ++index) {
                if (index > 0) {
                    std::cout << ' ';
                }
                std::cout << winners[neighbors[index] - 1];
            }
            std::cout << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}


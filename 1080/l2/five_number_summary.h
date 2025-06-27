#include <iostream>
#include <algorithm>
#include <stdint.h>

// Function to calculate the five-number summary
void fiveNumberSummary(uint64_t* data, size_t n) {
    // Step 1: Sort the data
    std::sort(data, data + n);

    // Step 2: Find the minimum value (first number in sorted data)
    uint64_t min = data[0];

    // Step 3: Find the maximum value (last number in sorted data)
    uint64_t max = data[n - 1];

    // Step 4: Find the median (Q2)
    uint64_t median = (n % 2 == 0) ? (data[n / 2 - 1] + data[n / 2]) / 2 : data[n / 2];

    // Step 5: Find the first quartile (Q1)
    uint64_t Q1 = (n / 2 % 2 == 0) ?
                    (data[(n / 2) / 2 - 1] + data[(n / 2) / 2]) / 2 :
                    data[(n / 2) / 2];

    // Step 6: Find the third quartile (Q3)
    uint64_t Q3 = ((n - n / 2) % 2 == 0) ?
                    (data[n / 2 + (n / 2) / 2 - 1] + data[n / 2 + (n / 2) / 2]) / 2 :
                    data[n / 2 + (n / 2) / 2];

    // Step 7: Output the five-number summary
    std::cout << "Five-Number Summary:" << std::endl;
    std::cout << "Minimum: " << min << std::endl;
    std::cout << "Q1 (First Quartile): " << Q1 << std::endl;
    std::cout << "Median (Q2): " << median << std::endl;
    std::cout << "Q3 (Third Quartile): " << Q3 << std::endl;
    std::cout << "Maximum: " << max << std::endl;
}

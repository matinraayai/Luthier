#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s) {
  size_t start = s.find_first_not_of(WHITESPACE);
  return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s) {
  size_t end = s.find_last_not_of(WHITESPACE);
  return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s) { return rtrim(ltrim(s)); }

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "an input file is required" << std::endl;
  }
  std::string filename = argv[1];
  std::string line;
  std::ifstream myfile(filename);
  std::ofstream outfile(filename + ".csv");
  if (myfile.is_open() && outfile.is_open()) {
    while (getline(myfile, line)) {
      if (line.find("//") != std::string::npos) {
        int location;
        location = line.find("//");
        std::string code = line.substr(0, location - 1);
        location = line.find(": ");
        std::string str_bytes_lo = line.substr(location + 2, 8);
        outfile << trim(code) << ";" << str_bytes_lo;
        // stringToByteArray(str_bytes_lo);
        if (line.size() > location + 10) {
          std::string str_bytes_hi = line.substr(location + 11, 8);
          outfile << ";" << str_bytes_hi;
          // stringToByteArray(str_bytes_hi);
        }
        outfile << std::endl;
      }
    }
    myfile.close();
    outfile.close();
  }

  else
    std::cout << "Unable to open file";

  return 0;
}
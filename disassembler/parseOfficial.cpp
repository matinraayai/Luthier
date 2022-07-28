#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
const std::string WHITESPACE = " \n\r\t\f\v";

std::string ltrim(const std::string &s)
{
	size_t start = s.find_first_not_of(WHITESPACE);
	return (start == std::string::npos) ? "" : s.substr(start);
}

std::string rtrim(const std::string &s)
{
	size_t end = s.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}

std::string trim(const std::string &s)
{
	return rtrim(ltrim(s));
}

std::vector<char> stringToByteArray(std::string str)
{

	const char *ptr = str.c_str();
	std::vector<char> bytes;
	for (int i = 0; i < 8; i += 2)
	{
		int j = 8 - 1 - i;
		uint8_t h4, l4;
		if (ptr[j - 1] < 65)
		{
			h4 = (uint8_t)(ptr[j - 1] - '0');
		}
		else
		{
			h4 = (uint8_t)(ptr[j - 1] - 'A' + 10);
		}
		if (ptr[j] < 65)
		{
			l4 = (uint8_t)(ptr[j] - '0');
		}
		else
		{
			l4 = (uint8_t)(ptr[j] - 'A' + 10);
		}

		uint32_t byte = (h4 << 4) + l4;
		bytes.push_back((char)byte);
		std::cout << std::hex << std::setw(2) << std::setfill('0') << byte;
	}
	std::cout << std::endl;
	return bytes;
}
int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cout << "an input file is required" << std::endl;
	}
	std::string filename = argv[1];
	std::string line;
	std::ifstream myfile(filename);
	std::ofstream outfile("official.csv");
	if (myfile.is_open() && outfile.is_open())
	{
		while (getline(myfile, line))
		{
			if (line.find("//") != std::string::npos)
			{
				int location;
				location = line.find("//");
				std::string code = line.substr(0, location - 1);
				location = line.find(": ");
				std::string str_bytes_lo = line.substr(location + 2, 8);
				outfile << trim(code) << ":" << str_bytes_lo;
				stringToByteArray(str_bytes_lo);
				if (line.size() > location + 10)
				{
					std::string str_bytes_hi = line.substr(location + 11, 8);
					outfile << str_bytes_hi;
					stringToByteArray(str_bytes_hi);
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
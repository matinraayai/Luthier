#include <iostream>
#include <fstream>
#include <string>
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

std::string reorderByteInWord(std::string bytes)
{

	const char *ptr = bytes.c_str();
	char ptr2[5];
	for (int i = 0; i < 4; i++)
	{
		ptr2[i] = ptr[3 - i];
	}
	ptr2[4] = '\0';
	std::string mystring(ptr2);
	return mystring;
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
				std::string bytes_lo = line.substr(location + 2, 8);

				std::cout << "first 4 bytes:" << reorderByteInWord(bytes_lo);
				if (line.size() > location + 10)
				{
					std::string bytes_hi = line.substr(location + 11, 8);
					std::cout << "second 4 bytes:" << reorderByteInWord(bytes_hi) << "end\n";
				}

				//outfile << trim(code) << ":" << bytes << "\n";
			}
		}
		myfile.close();
		outfile.close();
	}

	else
		std::cout << "Unable to open file";

	return 0;
}
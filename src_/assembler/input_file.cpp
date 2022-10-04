#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
	if (argc !=2) {
		printf("An input assembly file is required\n");
		return 1;
	}

	std::string filename = argv[1];
	std::string line;

	std::vector<std::string> instStr;
	//std::vector<const char*> instChar;
	std::vector<char*> instChar;
	

	std::ifstream file(filename, std::ios::in);
	if (file.is_open()){
		while(getline(file, line)) {
			//instStr.push_back(line);
			//instChar.push_back(reinterpret_cast<char*>(line.c_str()));
			instChar.push_back(line.c_str());
			std::cout << line.c_str() << std::endl;
		}
		file.close();
	} 
	/*
	for (int i = 0; i < instChar.size(); i++){
		std::cout << "String Vector:" << instStr.at(i) << std::endl;
		printf("Char Vector: %s\n", instChar.at(i));
	}
	*/

	return 0;
}

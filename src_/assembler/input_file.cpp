#include <fstream>
#include <iostream>
#include <string>
#include <vector>

std::string getInstName(std::string inst)
{
    int i = inst.find(" ");
    return inst.substr(0, i+1);
}

int main(int argc, char *argv[]) {
	if (argc !=2) {
		printf("An input assembly file is required\n");
		return 1;
	}

	std::string filename = argv[1];
	std::string line;

	std::vector<std::string> instvec;	

	std::ifstream file(filename, std::ios::in);
	if (file.is_open()){
		while(getline(file, line)) {
			instvec.push_back(line);
		}
		file.close();
	} 
	for (int i = 0; i < instvec.size(); i++){
		std::cout << "Inst name:" << getInstName(instvec.at(i)) << std::endl;
	}

	return 0;
}

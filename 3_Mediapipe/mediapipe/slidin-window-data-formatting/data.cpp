#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// #include <experimental/filesystem>

typedef long long ll;
#define debug(x) cerr<<#x<<": "<<x<<"\n";
using namespace std;

string filename = "new-demo-file.csv";
ofstream fout(filename);


vector<vector<string>> readcsv(string filename) {
    vector<vector<string>> data;
    ifstream file(filename);

    string line;
    while (getline(file, line)) {
        vector<string> row;
        //https://cplusplus.com/reference/sstream/stringstream/ 
        stringstream ss(line);

        string value;
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }
        data.push_back(row);
    }
    file.close();
    return data;
}

void writehead(vector<string> header, ll windowSize, string filename){
    
    //outputting the "class" column
    fout << header[0];
    for(int i=1;i<=windowSize;i++){
        for(int j=1;j<header.size();j++){
            fout  << ","<< i << header[j];
        }
    }
    fout<<"\n";
}

void writeline(vector<vector<string>> windowLine, ll windowSize, string filename){
    
    fout<<windowLine[0][0]; //outputting the "Class" column first.
    for(vector<string> line:windowLine){
        for(int i=1;i<line.size();i++){
            // debug(line[i])
            fout  << ","<< line[i];
        }
    }  
    fout<<"\n";

}

void writedata(vector<vector<string>> data, ll windowSize, string filename){

    writehead(data[0], windowSize, filename);

    //initializing the first window
    vector<vector<string>> windowLine;
    copy(data.begin()+1, data.begin()+(1+windowSize), back_inserter(windowLine));

    for(int i=2+windowSize; i<data.size();i++){
        writeline(windowLine, windowSize, filename);
        // windowLine.pop_front(); ==>doesn't exist in cpp, used for deque and lsits
        windowLine.erase(windowLine.begin());
        windowLine.push_back(data[i]);
        debug(i)
    }
}

// vector<string> getFilenamesInFolder(string folderPath) {
//     vector<string> filenames;
//     for (const auto &entry : filesystem::directory_iterator(folderPath)) {
//         filenames.push_back(entry.path().filename().string());
//     }
//     return filenames;
// }


int main(){
    // string filename = "new-demo-file.csv";
    ll windowSize = 10;
    vector<vector<string>> data = readcsv("demo-file.csv");

    
    writedata(data, windowSize, filename);
    
    return 0;
}

void test(vector<vector<string>> data){
    ofstream testout("new-demo-file.csv");
    for(auto x: data){
        for(auto y:x){
            testout<<y<<",";
        }
        testout<<"\n";
    }
}
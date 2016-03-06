#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <TooN/se3.h>


namespace se3_file_utils{

inline void readSE3nativefile(std::vector<TooN::SE3<> >& poses,
                              std::string& fileName)

{
    std::ifstream ifile(fileName.c_str());

    if ( !ifile.is_open() )
    {
        std::cerr << "File cannot be opened. Exiting.." << std::endl;
        ifile.close();
        exit(1);
    }

    char readlinedata[300];
    float val=0.0f;

    while(1)
    {
        ifile.getline(readlinedata,300);
        if ( ifile.eof() )
            break;

//        std::cout << readlinedata << std::endl;

        std::istringstream istring(readlinedata);

        TooN::Vector<6>se3_vector = TooN::Zeros(6);

        istring >> val;
        se3_vector[0]=val;

        istring >> val;
        se3_vector[1]=val;

        istring >> val;
        se3_vector[2]=val;

        istring >> val;
        se3_vector[3]=val;

        istring >> val;
        se3_vector[4]=val;

        istring >> val;
        se3_vector[5]=val;

        TooN::SE3<> tT_wc = TooN::SE3<>(se3_vector);

//        std::cout << "tT_wc pose = " << tT_wc << std::endl;

        poses.push_back(tT_wc);
    }

    ifile.close();
}

inline void readSE3MatrixRT(std::vector<TooN::SE3<> >& gtPoses,
                            std::string& fileName)
{
    std::ifstream ifile(fileName.c_str());

    if ( !ifile.is_open() )
    {
        std::cerr << "File cannot be opened. Exiting.." << std::endl;
        ifile.close();
        exit(1);
    }

//    char readlinedata[300];
//    float val=0.0f;

    TooN::SE3<>T_wc;

    while(1)
    {
//        ifile.getline(readlinedata,300);

        ifile >> T_wc;

        if ( ifile.eof() )
            break;

//        std::vector

        gtPoses.push_back(T_wc);
    }

    ifile.close();
}

}

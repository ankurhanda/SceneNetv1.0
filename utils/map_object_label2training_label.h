#include <iostream>
#include <string>






enum{
    UNKNOWN,// <0.4902,0.4824,0.4784>
    FLOOR,// <0,0,1.0000>
    CEILING,// <0.9137,0.3490,0.1882>
    WALL,// <0,0.8549,0>
    BED,// <0.5843,0,0.9412>
    CHAIR,// <0.8706,0.9451,0.0941>
    FURNITURE,// <1.0000,0.8078,0.8078>
    NIGHTSTAND,// <0,0.8784,0.8980>
    SHELF,// <0.4157,0.5333,0.8000>
    CURTAIN,// <0.4588,0.1137,0.1608>
    PAINTING,// <0.9412,0.1333,0.9216>
    PILLOWS,// <0,0.6549,0.6118>
    DOOR,// <0.9765,0.5451,0>
    WINDOW,// <0.8824,0.8980,0.7608>
    TABLE,// <1.0000,0,0>
    SOFA,// <0.8118,0.7176,0.2706>
    LAMP,// <0.7922,0.5804,0.5804>
    VASE,
    PLANT,
    PLATE,
    STAND
};

//% % NYU 14 classes
//% %          if index == 1% furniture
//% %             data.trainannot(hh,1,kk,jj) = 1;
//% %         elseif index == 2% object
//% %             data.trainannot(hh,2,kk,jj) = 1;
//% %         elseif index == 3% decoration
//% %             data.trainannot(hh,3,kk,jj) = 1;
//% %         elseif index == 4 || index == 0%unknown/padding
//% %             data.trainannot(hh,4,kk,jj) = 1;
//% %         elseif index == 5%books
//% %             data.trainannot(hh,5,kk,jj) = 1;
//% %         elseif index == 6% table
//% %             data.trainannot(hh,6,kk,jj) = 1;
//% %         elseif index == 7%sofa
//% %             data.trainannot(hh,7,kk,jj) = 1;
//% %         elseif index == 8%chair
//% %             data.trainannot(hh,8,kk,jj) = 1;
//% %         elseif index == 9%window
//% %             data.trainannot(hh,9,kk,jj) = 1;
//% %         elseif index == 10%television
//% %             data.trainannot(hh,10,kk,jj) = 1;
//% %         elseif index == 11%bed
//% %             data.trainannot(hh,11,kk,jj) = 1;
//% %         elseif index == 12%wall
//% %             data.trainannot(hh,12,kk,jj) = 1;
//% %         elseif index == 13%floor
//% %             data.trainannot(hh,13,kk,jj) = 1;
//% %         elseif index == 14%ceiling
//% %             data.trainannot(hh,14,kk,jj) = 1;
//% %          end

enum{
    FURNITURE_NYU,// <0,0,1.0000>
    OBJECT_NYU,// <0.9137,0.3490,0.1882>
    PAINTING_NYU,// <0,0.8549,0>
    UNKNOWN_NYU,// <0.5843,0,0.9412>
    BOOKS_NYU,// <0.8706,0.9451,0.0941>
    TABLE_NYU,// <1.0000,0.8078,0.8078>
    SOFA_NYU,// <0,0.8784,0.8980>
    CHAIR_NYU,// <0.4157,0.5333,0.8000>
    WINDOW_NYU,// <0.4588,0.1137,0.1608>
    TV_NYU,// <0.9412,0.1333,0.9216>
    BED_NYU,// <0,0.6549,0.6118>
    WALL_NYU,// <0.9765,0.5451,0>
    FLOOR_NYU,// <0.8824,0.8980,0.7608>
    CEILING_NYU
};


//enum{
//    FLOOR, // floor
//    CEILING, // ceiling
//    WALL, // wall
//    BED, // bed (duvet, sheets)
//    CHAIR, // chair
//    FURNITURE, // (cupboard, nightstand, chest-of-drawers, shelf)
//    CURTAIN,//curtains (blinds)
//    PAINTING,//painting (mirror)
//    PILLOWS,//pillows (cushions)
//    BOOKS,//books
//    DOOR,//door
//    WINDOW,//window
//    TABLE,//table (coffee-table)
//    SOFA, //sofa (bench)
//    UNKNOWN //unknown(decorations, lamps, clock, clothes, shoes, lights (switch), luminaire, vase, flowers, plants, computer monitor, glass, box)
//};


//    FLOOR: 0.34902, 0.552941, 0.94902
//    CEILING: 0.882353, 0.952941, 0.72549
//    WALL: 0.384314, 0.984314, 0.580392
//    BED: 0.0745098, 0.352941, 0.14902
//    CHAIR: 0.878431, 0.490196, 0.12549
//    FURNITURE: 0.0627451, 0.647059, 0.643137
//    CURTAIN: 0.909804, 0.709804, 0.466667
//    PAINTING: 0.466667, 0.313725, 0.945098
//    PILLOWS: 0.65098, 0.862745, 0.286275
//    BOOKS: 0.705882, 0.905882, 0.815686
//    DOOR: 0.133333, 0.0156863, 0.666667
//    WINDOW: 0.117647, 0.309804, 0.92549
//    TABLE: 0.788235, 0.905882, 0.737255
//    SOFA: 0.372549, 0.152941, 0.458824
//    UNKNOWN: 0.211765, 0.603922, 0.494118
//0.6, 0.482353, 0.235294
//0.678431, 0.305882, 0.0823529
//0.317647, 0.0196078, 0.290196
//0.231373, 0.329412, 0.0156863
//0.109804, 0.623529, 0.988235


std::string get_class_name(std::string& objectLabel)

{
    std::transform(objectLabel.begin(),
            objectLabel.end(),
            objectLabel.begin(),
            ::tolower);


    if ( ( objectLabel.find("floor") != std::string::npos )
            || objectLabel.find("carpet") != std::string::npos
            || objectLabel.find("ground") != std::string::npos )
        return "FLOOR";

    else if ( objectLabel.find("ceiling") != std::string::npos )
        return "CEILING";

    else if ( objectLabel.find("wall") != std::string::npos ||
            objectLabel.find("room_skeleton") != std::string::npos)
        return "WALL";

    else if (( objectLabel.find("bed") != std::string::npos)
            || ( objectLabel.find("duvet") != std::string::npos) )
        return "BED";

    else if ( objectLabel.find("chair") != std::string::npos)
        return "CHAIR";

    else if ( (objectLabel.find("cupboard") != std::string::npos ) ||
            (objectLabel.find("chest") != std::string::npos ) ||
            (objectLabel.find("drawers") != std::string::npos ) ||
            (objectLabel.find("furniture") != std::string::npos ) ||
            (objectLabel.find("bench") != std::string::npos ) ||
            (objectLabel.find("wardrobe") != std::string::npos ))
        return "FURNITURE";

    else if ( (objectLabel.find("nightstand") != std::string::npos ) ||
            (objectLabel.find("night_stand") != std::string::npos ) )
        return "NIGHTSTAND";

    else if ( (objectLabel.find("shelf") != std::string::npos ) ||
            (objectLabel.find("shelves") != std::string::npos ) )
        return "SHELF";

    else if ( ( objectLabel.find("curtain") != std::string::npos ) ||
            objectLabel.find("blind") != std::string::npos )
        return "CURTAIN";

    else if ( (objectLabel.find("painting") != std::string::npos ) ||
            (objectLabel.find("paint") != std::string::npos ) ||
            (objectLabel.find("mirror") != std::string::npos ) )
        return "PAINTING";

    else if ( (objectLabel.find("pillow") != std::string::npos ) ||
            (objectLabel.find("cushion") != std::string::npos ) )
        return "PILLOWS";

    //    else if ( (objectLabel.find("book") != std::string::npos )  )
    //        return "BOOKS";

    else if ( (objectLabel.find("door") != std::string::npos )  )
        return "DOOR";

    else if ( (objectLabel.find("window") != std::string::npos ) )
        return "WINDOW";

    else if ( objectLabel.find("table") != std::string::npos)
        return "TABLE";

    else if ( (objectLabel.find("sofa") != std::string::npos ) )
        return "SOFA";

    else if ( (objectLabel.find("lamp") != std::string::npos ) )
        return "LAMP";

    else if ( (objectLabel.find("palm") != std::string::npos ) )
        return "PLANT";

    else if ( (objectLabel.find("vase") != std::string::npos ) )
        return "VASE";

    else if ( (objectLabel.find("plate") != std::string::npos ) )
        return "PLATE";

    else
        return "UNKNOWN";
}

int obj_label2training_label_nyu_compatible(std::string& objectLabel)
{
    std::transform(objectLabel.begin(),
            objectLabel.end(),
            objectLabel.begin(),
            ::tolower);


    if ( ( objectLabel.find("floor")   != std::string::npos )
            || objectLabel.find("carpet") != std::string::npos
            || objectLabel.find("ground") != std::string::npos
            || objectLabel.find("rug")    != std::string::npos )
        return FLOOR_NYU;

    else if ( objectLabel.find("ceiling") != std::string::npos )
        return CEILING_NYU;

    else if ( objectLabel.find("wall") != std::string::npos
            || (objectLabel.find("door") != std::string::npos ) )
        return WALL_NYU;

    else if (( objectLabel.find("bed") != std::string::npos)
            || ( objectLabel.find("duvet") != std::string::npos)
            || (objectLabel.find("pillow") != std::string::npos ))
        return BED_NYU;

    else if ( objectLabel.find("chair") != std::string::npos)
        return CHAIR_NYU;

    else if ( (objectLabel.find("cupboard") != std::string::npos ) ||
            (objectLabel.find("chest") != std::string::npos ) ||
            (objectLabel.find("drawers") != std::string::npos ) ||
            (objectLabel.find("furniture") != std::string::npos ) ||
            (objectLabel.find("nightstand") != std::string::npos ) ||
            (objectLabel.find("night_stand") != std::string::npos ) ||
            (objectLabel.find("shelf") != std::string::npos ) ||
            (objectLabel.find("shelves") != std::string::npos ) )
        return FURNITURE_NYU;

    else if ( (objectLabel.find("painting") != std::string::npos ) ||
            (objectLabel.find("paint") != std::string::npos ) ||
            (objectLabel.find("mirror") != std::string::npos ) ||
            (objectLabel.find("tv") != std::string::npos ) )
        return PAINTING_NYU;


    else if ( (objectLabel.find("window") != std::string::npos )
            || ( objectLabel.find("curtain") != std::string::npos ) ||
            objectLabel.find("blind") != std::string::npos)
        return WINDOW_NYU;

    else if ( objectLabel.find("table") != std::string::npos)
        return TABLE_NYU;

    else if ( (objectLabel.find("sofa") != std::string::npos ) )
        return SOFA_NYU;

    else
        return OBJECT_NYU;

}



int obj_label2training_label(std::string& objectLabel)
{
    std::transform(objectLabel.begin(),
            objectLabel.end(),
            objectLabel.begin(),
            ::tolower);


    if ( ( objectLabel.find("floor") != std::string::npos )
            || objectLabel.find("carpet") != std::string::npos
            || objectLabel.find("ground") != std::string::npos
            || objectLabel.find("rug") != std::string::npos
            || objectLabel.find("mat_") != std::string::npos )
        return FLOOR;

    else if ( objectLabel.find("ceiling") != std::string::npos )
        return CEILING;

    else if ( objectLabel.find("wall") != std::string::npos||
            objectLabel.find("room_skeleton") != std::string::npos)
        return WALL;

    else if (( objectLabel.find("bed") != std::string::npos)
            || ( objectLabel.find("duvet") != std::string::npos)
            || (objectLabel.find("furniture") != std::string::npos )
            )
        return BED;

    else if ( objectLabel.find("chair") != std::string::npos)
        return CHAIR;

    else if ( (objectLabel.find("cupboard") != std::string::npos ) ||
            (objectLabel.find("bench") != std::string::npos ) ||
            //              (objectLabel.find("furniture") != std::string::npos )||
            (objectLabel.find("chest") != std::string::npos ) ||
            (objectLabel.find("drawers") != std::string::npos ) ||
            (objectLabel.find("wardrobe") != std::string::npos ))
        return FURNITURE;

    else if ( (objectLabel.find("nightstand") != std::string::npos ) ||
            (objectLabel.find("night_stand") != std::string::npos ) )
        return NIGHTSTAND;

    else if ( (objectLabel.find("shelf") != std::string::npos ) ||
            (objectLabel.find("shelves") != std::string::npos ) )
        return SHELF;

    else if ( ( objectLabel.find("curtain") != std::string::npos ) ||
            objectLabel.find("blind") != std::string::npos )
        return CURTAIN;

    else if ( (objectLabel.find("painting") != std::string::npos ) ||
            (objectLabel.find("paint") != std::string::npos ) ||
            (objectLabel.find("mirror") != std::string::npos ) ||
            (objectLabel.find("tv") != std::string::npos ) )
        return PAINTING;

    else if ( (objectLabel.find("pillow") != std::string::npos ) ||
            (objectLabel.find("cushion") != std::string::npos ) ||
            (objectLabel.find("pilow") != std::string::npos ) )
        return PILLOWS;

    //    else if ( (objectLabel.find("book") != std::string::npos )  )
    //        return "BOOKS";

    //    else if ( (objectLabel.find("door") != std::string::npos )  )
    //        return DOOR;

    /// Just putting door into windows for now...
    else if ( (objectLabel.find("window") != std::string::npos )
            || (objectLabel.find("door") != std::string::npos ))
        return WINDOW;

    else if ( objectLabel.find("table") != std::string::npos)
        return TABLE;

    else if ( (objectLabel.find("sofa") != std::string::npos ) )
        return SOFA;

    else if ( (objectLabel.find("lamp") != std::string::npos ) )
        return LAMP;

    else if ( (objectLabel.find("palm") != std::string::npos ) )
        return PLANT;

    else if ( (objectLabel.find("vase") != std::string::npos ) )
        return VASE;

    else if ( (objectLabel.find("plate") != std::string::npos ) )
        return PLATE;

    else if ( (objectLabel.find("stand") != std::string::npos ) )
        return STAND;

    else
        return UNKNOWN;

}

std::string index2class_name(int index)
{
    switch (index)
    {
        case 0:
            return "UNKNOWN";
        case 1:
            return "FLOOR"; // floor
        case 2:
            return  "CEILING"; // ceiling
        case 3:
            return "WALL"; // wall
        case 4:
            return "BED"; // bed (duvet; sheets)
        case 5:
            return "CHAIR"; // chair
        case 6:
            return "FURNITURE"; // (cupboard; nightstand; chest-of-drawers; shelf)
        case 7:
            return "CURTAIN";//curtains (blinds)
        case 8:
            return "PAINTING";//painting (mirror)
        case 9:
            return "PILLOWS";//pillows (cushions)
        case 10:
            return "BOOKS";//books
        case 11:
            return "DOOR";//door
        case 12:
            return "WINDOW";//window
        case 13:
            return "TABLE";//table (coffee-table)
        case 14:
            return "SOFA"; //sofa (bench)
        case 15:
            return "LAMP";
        case 16:
            return "VASE";
        case 17:
            return "PLANT";
        case 18:
            return "PLATE";
        case 19:
            return "STAND";
    }
}

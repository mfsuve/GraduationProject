#include <iostream>
#include <string>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

#define PType pcl::PointXYZRGBA

std::string name;

class SimpleOpenNIViewer {
    public:

        pcl::Grabber* interface;

        SimpleOpenNIViewer() : interface(new pcl::OpenNIGrabber()) {}

        void cloud_cb_ (const pcl::PointCloud<PType>::ConstPtr &cloud) {
            std::string path("../../data/" + name + "/" + name + "2.pcd");
            std::cout << "Path: " << path << std::endl;
            pcl::io::savePCDFileASCII (path, *cloud);
            interface->stop ();
        }

        void run () {
            boost::function<void (const pcl::PointCloud<PType>::ConstPtr&)> f = boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

            interface->registerCallback (f);

            interface->start ();
        }
};

int main (int argc, char* argv[]) {
    SimpleOpenNIViewer v;
    if(argc != 2) {
        std::cout << "Please enter one name" << std:: endl;
        return 0;
    }
    name = std::string(argv[1]);
    std::cout << "Name: " << name << std::endl;
    v.run ();
    return 0;
}

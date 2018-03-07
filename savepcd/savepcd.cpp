#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

#define PType pcl::PointXYZRGBA

class SimpleOpenNIViewer {
    public:

        pcl::Grabber* interface;

        SimpleOpenNIViewer() : interface(new pcl::OpenNIGrabber()) {}

        void cloud_cb_ (const pcl::PointCloud<PType>::ConstPtr &cloud) {
            pcl::io::savePCDFileASCII ("../../testdata/sayitut/sayitut.pcd", *cloud);
            interface->stop ();
        }

        void run () {
            boost::function<void (const pcl::PointCloud<PType>::ConstPtr&)> f = boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

            interface->registerCallback (f);

            interface->start ();
        }
};

int main () {
    SimpleOpenNIViewer v;
    v.run ();
    return 0;
}

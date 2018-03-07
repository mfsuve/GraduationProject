#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#define PType pcl::PointXYZRGBA

class SimpleOpenNIProcessor {
public:

	SimpleOpenNIProcessor () : viewer ("PCL OpenNI Viewer") {} 

        boost::shared_ptr<pcl::visualization::PCLVisualizer> rgbVis (pcl::PointCloud<PType>::ConstPtr cloud)
        {
          // --------------------------------------------
          // -----Open 3D viewer and add point cloud-----
          // --------------------------------------------
          boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
          viewer->setBackgroundColor (0, 0, 0);
          pcl::visualization::PointCloudColorHandlerRGBField<PType> rgb(cloud);
          viewer->addPointCloud<PType> (cloud, rgb, "sample cloud");
          viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
          viewer->addCoordinateSystem (1.0);
          viewer->initCameraParameters ();
          return (viewer);
        }

        void cloud_cb_ (const pcl::PointCloud<PType>::ConstPtr &cloud) {
            // FILTERING
            pcl::PointCloud<PType>::Ptr cloud_filtered (new pcl::PointCloud<PType>);

            double center_distance = cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z;

            std::cout << "center_distance: " << center_distance << std::endl;

            pcl::PassThrough<PType> pass;
            pass.setInputCloud (cloud);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (center_distance - 0.2, center_distance + 0.2);
            //pass.setFilterLimitsNegative (true);
            pass.filter (*cloud_filtered);


            // SEGMENTATION
            // you can try euclidian segmentation to eliminate the points that are not of the books but in the same plane
            pcl::PointCloud<PType>::Ptr final (new pcl::PointCloud<PType>);

            pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // Create the segmentation object
            pcl::SACSegmentation<PType> seg;
            // Optional
            seg.setOptimizeCoefficients (true);
            // Mandatory
            seg.setModelType (pcl::SACMODEL_PLANE);
            seg.setMethodType (pcl::SAC_RANSAC);
            seg.setDistanceThreshold (0.01);

            seg.setInputCloud (cloud_filtered);
            seg.segment (*inliers, *coefficients);
/*
            std::cerr << "Model coefficients: " << coefficients->values[0] << " "
                                                << coefficients->values[1] << " "
                                                << coefficients->values[2] << " "
                                                << coefficients->values[3] << std::endl;
*/
            pcl::copyPointCloud<PType>(*cloud_filtered, inliers->indices, *final);

            // creates the visualization object and adds either our orignial cloud or all of the inliers
            // depending on the command line arguments specified.
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
            viewer = rgbVis(final);
            //while (!viewer->wasStopped ()) {
              viewer->spinOnce (100);
              boost::this_thread::sleep (boost::posix_time::seconds (150));
            //}

//		if (!viewer.wasStopped())
//			viewer.showCloud (cloud);
	}

	void run ()	{
		// create a new grabber for OpenNI devices
		pcl::Grabber* interface = new pcl::OpenNIGrabber();

		// make callback function from member function
                boost::function<void (const pcl::PointCloud<PType>::ConstPtr&)> f =
			boost::bind (&SimpleOpenNIProcessor::cloud_cb_, this, _1);

		// connect callback function for desired signal. In this case its a point cloud with color values
		boost::signals2::connection c = interface->registerCallback (f);

		// start receiving point clouds
		interface->start ();

                // wait until user quits program with Ctrl-C, but no busy-waiting -> sleep (1);

                //while (!viewer.wasStopped())
                        boost::this_thread::sleep (boost::posix_time::seconds (150));

		// stop the grabber
                cout << "Interface stopped!" << endl;
		interface->stop ();
	}
	
	pcl::visualization::CloudViewer viewer;	

};

int main () {
	SimpleOpenNIProcessor v;
	v.run ();
	return (0);
}

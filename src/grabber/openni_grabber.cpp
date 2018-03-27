#include <iostream>
#include <string>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
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

#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pngwriter.h>

#define PType pcl::PointXYZRGBA


class SimpleOpenNIViewer {
public:
	SimpleOpenNIViewer () : interface(new pcl::OpenNIGrabber()) {}

	void cloud_cb_ (const pcl::PointCloud<PType>::ConstPtr &cloud) {
		interface->stop ();

		int W = cloud->width, H = cloud->height;

		pcl::PointCloud<PType>::Ptr final (new pcl::PointCloud<PType>);

		// FILTERING
		pcl::PointCloud<PType>::Ptr cloud_filtered (new pcl::PointCloud<PType>);
		std::vector<int> distance_filter_indices;

		pcl::PassThrough<PType> pass;
		pass.setInputCloud (cloud);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.2, 1.2);
		pass.filter (distance_filter_indices);

		pcl::copyPointCloud<PType>(*cloud, distance_filter_indices, *cloud_filtered);

		// SEGMENTATION
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

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<PType> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Get the points associated with the planar surface
		extract.filter (*final);

		// EUCLIDEAN CLUSTER SEGMENTATION
		// Creating the KdTree object for the search method of the extraction
		pcl::search::KdTree<PType>::Ptr tree (new pcl::search::KdTree<PType>);
		tree->setInputCloud (final);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<PType> ec;
		ec.setClusterTolerance (0.005); // 2cm
		ec.setMinClusterSize (100);
		ec.setMaxClusterSize (50000);
		ec.setSearchMethod (tree);
		ec.setInputCloud (final);
		ec.extract (cluster_indices);

		//std::cout << std::endl << "cluster_indices.size(): " << cluster_indices.size() << std::endl;

		int count = 0;
		pcl::PointIndices::Ptr maxCluster (new pcl::PointIndices);
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
			//std::cout << ++count << ". Cluster->" << "size(): " << it->indices.size() << std::endl;
			if(maxCluster->indices.size() < it->indices.size())
				maxCluster->indices = it->indices;
		}

		pcl::PointCloud<PType>::Ptr cloud_cluster (new pcl::PointCloud<PType>);
		for (std::vector<int>::const_iterator pit = maxCluster->indices.begin (); pit != maxCluster->indices.end (); ++pit)
			cloud_cluster->points.push_back (final->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		W = cloud->width;
		H = cloud->height;
		pngwriter image2(W, H, 0.0, "../../python/finals/temp/cloud_image.png");
		for(int ii = 0; ii < maxCluster->indices.size(); ii++) {
			int index = distance_filter_indices[inliers->indices[maxCluster->indices[ii]]];
			int i = index % W;
			int j = index / W;
			PType p = cloud->points[index];
			image2.plot(i, H - j - 1, p.r / 256.0, p.g / 256.0, p.b / 256.0);
		}
		image2.close();

/*
		pcl::visualization::CloudViewer viewer ("PCL OpenNI Viewer");
		viewer.showCloud (cloud);
		while (!viewer.wasStopped ()) {
			boost::this_thread::sleep (boost::posix_time::seconds (1));
		}
*/

		boost::this_thread::sleep (boost::posix_time::seconds (5));
		std::cout << "Please press Enter..." << std::endl;
		std::cin.get();
		interface->start ();
	}

	void run () {

		boost::function<void (const pcl::PointCloud<PType>::ConstPtr&)> f =
			boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

		interface->registerCallback (f);

		interface->start ();

		boost::this_thread::sleep (boost::posix_time::seconds (500));
	}

	pcl::Grabber* interface;

};

int main () {
	SimpleOpenNIViewer v;
	v.run ();
	return 0;
}
















/*

class SimpleOpenNIViewer {
public:
	SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}

	void cloud_cb_ (const pcl::PointCloud<PType>::ConstPtr &cloud) {
		interface->stop ();
		std::cin.get();



		int W = cloud->width, H = cloud->height;

		pcl::visualization::CloudViewer viewer("Cloud Viewer");

		pcl::PointCloud<PType>::Ptr final (new pcl::PointCloud<PType>);

		// FILTERING
		pcl::PointCloud<PType>::Ptr cloud_filtered (new pcl::PointCloud<PType>);
		std::vector<int> distance_filter_indices;

		pcl::PassThrough<PType> pass;
		pass.setInputCloud (cloud);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (0.2, 1.2);
		pass.filter (distance_filter_indices);

		pcl::copyPointCloud<PType>(*cloud, distance_filter_indices, *cloud_filtered);

		// SEGMENTATION
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

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<PType> extract;
		extract.setInputCloud (cloud_filtered);
		extract.setIndices (inliers);
		extract.setNegative (false);

		// Get the points associated with the planar surface
		extract.filter (*final);

		// EUCLIDEAN CLUSTER SEGMENTATION
		// Creating the KdTree object for the search method of the extraction
		pcl::search::KdTree<PType>::Ptr tree (new pcl::search::KdTree<PType>);
		tree->setInputCloud (final);

		std::vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<PType> ec;
		ec.setClusterTolerance (0.005); // 2cm
		ec.setMinClusterSize (100);
		ec.setMaxClusterSize (50000);
		ec.setSearchMethod (tree);
		ec.setInputCloud (final);
		ec.extract (cluster_indices);

		std::cout << std::endl << "cluster_indices.size(): " << cluster_indices.size() << std::endl;

		int count = 0;
		pcl::PointIndices::Ptr maxCluster (new pcl::PointIndices);
		for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
			std::cout << ++count << ". Cluster->" << "size(): " << it->indices.size() << std::endl;
			if(maxCluster->indices.size() < it->indices.size())
				maxCluster->indices = it->indices;
		}

		pcl::PointCloud<PType>::Ptr cloud_cluster (new pcl::PointCloud<PType>);
		for (std::vector<int>::const_iterator pit = maxCluster->indices.begin (); pit != maxCluster->indices.end (); ++pit)
			cloud_cluster->points.push_back (final->points[*pit]); //*
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		W = cloud->width;
		H = cloud->height;
		pngwriter image2(W, H, 0.0, "../../python/finals/temp/cloud_image.png");
		for(int ii = 0; ii < maxCluster->indices.size(); ii++) {
			int index = distance_filter_indices[inliers->indices[maxCluster->indices[ii]]];
			int i = index % W;
			int j = index / W;
			PType p = cloud->points[index];
			image2.plot(i, j, p.r / 256.0, p.g / 256.0, p.b / 256.0);
		}
		image2.close();



		interface->start ();
	}

	void run () {
		interface = new pcl::OpenNIGrabber();

		boost::function<void (const pcl::PointCloud<pcl::PointXYZ>::ConstPtr&)> f =
			boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

		interface->registerCallback (f);

		interface->start ();

		while (!viewer.wasStopped()) {
			boost::this_thread::sleep (boost::posix_time::seconds (1));
		}

	}

	pcl::visualization::CloudViewer viewer;
	pcl::Grabber* interface;
};

int main () {
	SimpleOpenNIViewer v;
	v.run ();
	return 0;
}


*/





































/*
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

            pcl::PointCloud<PType>::Ptr final (new pcl::PointCloud<PType>);

            // FILTERING
            pcl::PointCloud<PType>::Ptr cloud_filtered (new pcl::PointCloud<PType>);
            std::vector<int> distance_filter_indices;

            pcl::PassThrough<PType> pass;
            pass.setInputCloud (cloud);
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (0.2, 0.8);
            pass.filter (distance_filter_indices);

            pcl::copyPointCloud<PType>(*cloud, distance_filter_indices, *cloud_filtered);

            // SEGMENTATION
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

            // Extract the planar inliers from the input cloud
            pcl::ExtractIndices<PType> extract;
            extract.setInputCloud (cloud_filtered);
            extract.setIndices (inliers);
            extract.setNegative (false);

            // Get the points associated with the planar surface
            extract.filter (*final);

            // EUCLIDEAN CLUSTER SEGMENTATION
            // Creating the KdTree object for the search method of the extraction
            pcl::search::KdTree<PType>::Ptr tree (new pcl::search::KdTree<PType>);
            tree->setInputCloud (final);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<PType> ec;
            ec.setClusterTolerance (0.005); // 2cm
            ec.setMinClusterSize (100);
            ec.setMaxClusterSize (50000);
            ec.setSearchMethod (tree);
            ec.setInputCloud (final);
            ec.extract (cluster_indices);

            int count = 0;
            pcl::PointIndices::Ptr maxCluster (new pcl::PointIndices);
            for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
                if(maxCluster->indices.size() < it->indices.size())
                    maxCluster->indices = it->indices;
            }

            pcl::PointCloud<PType>::Ptr cloud_cluster (new pcl::PointCloud<PType>);
            for (std::vector<int>::const_iterator pit = maxCluster->indices.begin (); pit != maxCluster->indices.end (); ++pit)
                cloud_cluster->points.push_back (final->points[*pit]);
            cloud_cluster->width = cloud_cluster->points.size ();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            if (!viewer.wasStopped())
              viewer.showCloud (cloud_cluster);
	}

        void run () {
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

            while (!viewer.wasStopped())
                boost::this_thread::sleep (boost::posix_time::seconds (1));

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
*/

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

std::string png_name;
std::string pcd_name;

void parse_arguments(int argc, char* argv[]) {
    std::string cloud_name("hayvan");
    std::string i("1");
    bool test = false;
    if(argc > 1)
        cloud_name = std::string(argv[1]);
    if(argc > 2)
        if(std::string(argv[2]).compare("test") != 0)
            i = std::string(argv[2]);
        else
            test = true;
    if(argc > 3)
        test = true;

    if(test) {
        pcd_name = "../../testdata/" + cloud_name + "/" + cloud_name + ".pcd";
        png_name = "png_files/" + cloud_name + "_test.png";
    }
    else {
        png_name = "png_files/" + cloud_name + i + ".png";
        pcd_name = "../../data/" + cloud_name + "/" + cloud_name + i + ".pcd";
    }
}

int main (int argc, char* argv[]) {

    parse_arguments(argc, argv);

    pcl::PointCloud<PType>::Ptr cloud (new pcl::PointCloud<PType>);
    pcl::io::loadPCDFile (pcd_name.c_str(), *cloud);

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
    pngwriter image2(W, H, 0.0, png_name.c_str());
    for(int ii = 0; ii < maxCluster->indices.size(); ii++) {
        int index = distance_filter_indices[inliers->indices[maxCluster->indices[ii]]];
        int i = index % W;
        int j = index / W;
        PType p = cloud->points[index];
        image2.plot(i, j, p.r / 256.0, p.g / 256.0, p.b / 256.0);
    }
    image2.close();

    viewer.showCloud(final);

    while (!viewer.wasStopped ()) {
        boost::this_thread::sleep (boost::posix_time::seconds (1));
    }

    return 0;
}
